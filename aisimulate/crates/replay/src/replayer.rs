// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public replay facade over the mechanically moved topology runtimes.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result as AnyResult;
use uuid::Uuid;

use crate::agg::AggRuntimeImpl;
use crate::artifact::{ReplayArtifactKvEventVisibility, ReplayArtifactSink, ReplayArtifacts};
use crate::components::{
    AdmissionQueue, NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::core::round_robin::{AggregatedRoundRobinPlacement, PoolRoundRobinPlacement};
use crate::core::{NoEngineEvents, PlacementPolicy, WorkerTopology};
use crate::disagg::DisaggRuntimeImpl;
use crate::loadgen::ReplayRequestPayload;
use crate::loadgen::WorkloadDriver;
use crate::native::{NativeReplayEngineConfig, NativeReplayFactory};
use crate::protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};
use crate::replay::OfflineDisaggReplayConfig;
use crate::scaling::ReplayScalingPolicy;
use crate::single::{SingleReplayMode, SingleRuntime};
use crate::{
    ReplayError, ReplayResult, ReplaySpec, ReplayTopology, SlaThresholds, TraceSimulationReport,
    WorkerStage,
};

/// Runtime composition supplied by the built-in engine stack or a Dynamo
/// adapter. The adapter owns concrete Router/Planner construction; Replay only
/// sees the already-neutral placement and scaling contracts.
pub trait ReplayComposition {
    type Metadata: ReplayAdmissionMetadata;
    type Observation: ReplayEngineObservation;
    type AggregatedPlacement: PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Self::Metadata,
            Observation = <Self::Observation as ReplayEngineObservation>::Batch,
        >;
    type DisaggregatedPlacement: PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Self::Metadata,
            Observation = <Self::Observation as ReplayEngineObservation>::Batch,
        >;

    fn validate_spec(&self, _spec: &ReplaySpec) -> ReplayResult<()> {
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> AnyResult<Self::AggregatedPlacement>;

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> AnyResult<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)>;

    /// Return the run-owned scaling policy, if this composition has one.
    fn take_scaling_policy(&mut self) -> ReplayResult<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(None)
    }

    /// Only the built-in Round-robin composition may bypass the generic
    /// placement loop for the preserved single-worker fast path.
    fn supports_single_worker_runtime(&self) -> bool {
        false
    }
}

/// Replay-owned runtime input used by compatibility runners that already
/// lowered a trace into the shared workload driver.
///
/// Serializable callers should keep using [`ReplaySpec::requests`]. Dynamo's
/// legacy entrypoints use this seam to preserve multi-turn, concurrency, and
/// agentic scheduling without recompiling Replay sources in the Dynamo crate.
#[doc(hidden)]
pub enum ReplayRuntimeInput {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

/// Built-in engine-only composition: Round-robin placement and fixed capacity.
#[derive(Debug, Default, Clone, Copy)]
pub struct RoundRobinComposition;

impl ReplayComposition for RoundRobinComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn validate_spec(&self, spec: &ReplaySpec) -> ReplayResult<()> {
        if spec.adapters.placement.provider != "round_robin" {
            return Err(ReplayError::InvalidSpec(format!(
                "engine composition requires round_robin placement, got {:?}",
                spec.adapters.placement.provider
            )));
        }
        if spec.adapters.scaling.provider != "none" {
            return Err(ReplayError::InvalidSpec(format!(
                "engine composition does not provide scaling, got {:?}",
                spec.adapters.scaling.provider
            )));
        }
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> AnyResult<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> AnyResult<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn supports_single_worker_runtime(&self) -> bool {
        true
    }
}

/// Owns one replay execution: canonical spec, native engine construction, and
/// the selected placement/scaling composition.
pub struct Replayer<C = RoundRobinComposition> {
    spec: ReplaySpec,
    factory: NativeReplayFactory,
    composition: C,
    runtime_input: Option<ReplayRuntimeInput>,
}

impl Replayer<RoundRobinComposition> {
    pub fn new(spec: ReplaySpec, factory: NativeReplayFactory) -> ReplayResult<Self> {
        Self::with_composition(spec, factory, RoundRobinComposition)
    }

    /// Run a fixed, aggregated single-worker replay and retain detailed
    /// request/output/native-KV observations from the same Replayer-owned
    /// virtual-clock loop.
    ///
    /// This contract intentionally targets one worker artifact. Multi-worker,
    /// scaling, and disaggregated runs should consume the normal report and
    /// placement/scaling observation contracts instead of creating a second
    /// scheduler loop solely for artifact generation.
    pub fn run_with_artifacts(
        self,
        visibility: ReplayArtifactKvEventVisibility,
    ) -> ReplayResult<(TraceSimulationReport, ReplayArtifacts)> {
        let sink = ReplayArtifactSink::new(visibility);
        let report = self.run_inner(Some(sink.clone()))?;
        Ok((report, sink.take()?))
    }
}

impl<C: ReplayComposition> Replayer<C> {
    pub fn with_composition(
        spec: ReplaySpec,
        factory: NativeReplayFactory,
        composition: C,
    ) -> ReplayResult<Self> {
        spec.validate()?;
        composition.validate_spec(&spec)?;
        Ok(Self {
            spec,
            factory,
            composition,
            runtime_input: None,
        })
    }

    /// Override the serializable request list with an already-lowered,
    /// Replay-owned runtime input.
    #[doc(hidden)]
    pub fn with_runtime_input(mut self, input: ReplayRuntimeInput) -> Self {
        self.runtime_input = Some(input);
        self
    }

    pub fn run(self) -> ReplayResult<TraceSimulationReport> {
        self.run_inner(None)
    }

    fn run_inner(
        mut self,
        artifact_sink: Option<ReplayArtifactSink>,
    ) -> ReplayResult<TraceSimulationReport> {
        let wall_start = Instant::now();
        let engine_config = NativeReplayEngineConfig::parse(&self.spec.engine)?;
        let runtime_input = match self.runtime_input.take() {
            Some(input) => input,
            None => ReplayRuntimeInput::Requests(lower_requests(&self.spec)?),
        };
        let mode = self
            .spec
            .max_in_flight
            .map_or(ReplayMode::Trace, |max_in_flight| ReplayMode::Concurrency {
                max_in_flight,
            });
        let scaling = self.composition.take_scaling_policy()?;

        let collector = match &self.spec.topology {
            ReplayTopology::Aggregated { workers } => {
                let role_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Aggregated,
                    C::Observation::CAPTURE_NATIVE_KV_EVENTS || artifact_sink.is_some(),
                )?;
                let startup_time_ms = positive_delay(workers.startup_delay_ms);

                let use_single_worker_runtime = workers.initial_workers == 1
                    && role_factory.dp_size() == 1
                    && scaling.is_none()
                    && self.composition.supports_single_worker_runtime();
                if artifact_sink.is_some() && !use_single_worker_runtime {
                    return Err(ReplayError::InvalidSpec(
                        "detailed replay artifacts require fixed aggregated topology with one logical DP1 worker"
                            .to_string(),
                    ));
                }

                if use_single_worker_runtime {
                    let single_mode = match mode {
                        ReplayMode::Trace => SingleReplayMode::Trace,
                        ReplayMode::Concurrency { max_in_flight } => {
                            SingleReplayMode::Concurrency { max_in_flight }
                        }
                    };
                    let runtime = match runtime_input {
                        ReplayRuntimeInput::Requests(pending) => {
                            SingleRuntime::new(role_factory, pending, single_mode)
                        }
                        ReplayRuntimeInput::Workload(driver) => {
                            SingleRuntime::new_workload(role_factory, driver, single_mode)
                        }
                    };
                    let mut runtime = runtime
                        .map_err(runtime_error)?
                        .with_per_request_records(self.spec.record_per_request)
                        .with_max_sim_time_ms(self.spec.max_sim_time_ms);
                    if let Some(sink) = artifact_sink {
                        runtime = runtime.with_artifact_sink(sink);
                    }
                    runtime.run().map_err(runtime_error)?
                } else {
                    let mut runtime = AggRuntimeImpl::<
                        C::AggregatedPlacement,
                        C::Observation,
                        C::Metadata,
                    >::new_composed(
                        role_factory,
                        admission_queue(runtime_input, mode),
                        workers.initial_workers,
                        startup_time_ms,
                        |dp_size, topology| {
                            self.composition
                                .create_aggregated_placement(dp_size, topology)
                        },
                    )
                    .map_err(runtime_error)?
                    .with_per_request_records(self.spec.record_per_request)
                    .with_max_sim_time_ms(self.spec.max_sim_time_ms);
                    if let Some(policy) = scaling {
                        runtime = runtime.with_scaling_policy(policy);
                    }
                    runtime.run().map_err(runtime_error)?.0
                }
            }
            ReplayTopology::Disaggregated {
                prefill,
                decode,
                handoff_latency_ms,
            } => {
                if artifact_sink.is_some() {
                    return Err(ReplayError::InvalidSpec(
                        "detailed replay artifacts require aggregated topology".to_string(),
                    ));
                }
                let prefill_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Prefill,
                    C::Observation::CAPTURE_NATIVE_KV_EVENTS,
                )?;
                let decode_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Decode,
                    C::Observation::CAPTURE_NATIVE_KV_EVENTS,
                )?;
                let config = OfflineDisaggReplayConfig {
                    prefill_factory,
                    decode_factory,
                    prefill_startup_time_ms: positive_delay(prefill.startup_delay_ms),
                    decode_startup_time_ms: positive_delay(decode.startup_delay_ms),
                    num_prefill_workers: prefill.initial_workers,
                    num_decode_workers: decode.initial_workers,
                    handoff_latency_ms: *handoff_latency_ms,
                };
                let mut runtime = DisaggRuntimeImpl::<
                    C::DisaggregatedPlacement,
                    C::Observation,
                    C::Metadata,
                >::new_composed(
                    &config,
                    admission_queue(runtime_input, mode),
                    false,
                    |prefill_dp, prefill_topology, decode_dp, decode_topology| {
                        self.composition.create_disaggregated_placements(
                            prefill_dp,
                            prefill_topology,
                            decode_dp,
                            decode_topology,
                        )
                    },
                )
                .map_err(runtime_error)?
                .with_per_request_records(self.spec.record_per_request)
                .with_max_sim_time_ms(self.spec.max_sim_time_ms);
                if let Some(policy) = scaling {
                    runtime = runtime.with_scaling_policy(policy);
                }
                runtime.run().map_err(runtime_error)?.0
            }
        };

        Ok(finish_report(collector, self.spec.sla)
            .with_wall_time_ms(wall_start.elapsed().as_secs_f64() * 1_000.0))
    }
}

fn admission_queue<Metadata: ReplayAdmissionMetadata>(
    input: ReplayRuntimeInput,
    mode: ReplayMode,
) -> AdmissionQueue<Metadata> {
    match input {
        ReplayRuntimeInput::Requests(requests) => AdmissionQueue::new_requests(requests, mode),
        ReplayRuntimeInput::Workload(driver) => AdmissionQueue::new_workload(driver, mode),
    }
}

fn positive_delay(delay_ms: f64) -> Option<f64> {
    (delay_ms > 0.0).then_some(delay_ms)
}

fn lower_requests(spec: &ReplaySpec) -> ReplayResult<VecDeque<DirectRequest>> {
    let mut pending = spec
        .requests
        .iter()
        .enumerate()
        .map(|(index, request)| -> ReplayResult<_> {
            let request_id = Uuid::from_u128(
                u128::try_from(index)
                    .expect("usize always fits u128")
                    .checked_add(1)
                    .expect("replay request index overflow"),
            );
            let (tokens, prompt_token_source) = match &request.input_token_ids {
                Some(tokens) => (tokens.clone(), ReplayPromptTokenSource::Materialized),
                None => {
                    let seed = u32::try_from(index)
                        .unwrap_or(u32::MAX)
                        .wrapping_mul(1_000_003);
                    (
                        (0..request.input_tokens)
                            .map(|offset| {
                                seed.wrapping_add(u32::try_from(offset).unwrap_or(u32::MAX))
                            })
                            .collect(),
                        ReplayPromptTokenSource::LengthOnlySynthetic,
                    )
                }
            };
            let routing = request.routing_metadata()?;
            Ok(DirectRequest {
                tokens,
                max_output_tokens: request.output_tokens,
                output_token_ids: request.output_token_ids.clone(),
                uuid: Some(request_id),
                dp_rank: 0,
                preferred_dp_rank: request.dp_rank,
                arrival_timestamp_ms: Some(request.arrival_time_ms),
                priority: routing.priority,
                strict_priority: routing.strict_priority,
                policy_class: routing.policy_class,
                replay_context: Some(ReplayRequestContext {
                    authored_id: request.id.clone(),
                    session_id: request.session_id.clone(),
                    turn_index: request.turn_index,
                    metadata: request.metadata.clone(),
                    prompt_token_source,
                }),
            })
        })
        .collect::<ReplayResult<Vec<_>>>()?;
    pending.sort_by(|left, right| {
        left.arrival_timestamp_ms
            .expect("ReplaySpec request always has an arrival")
            .total_cmp(
                &right
                    .arrival_timestamp_ms
                    .expect("ReplaySpec request always has an arrival"),
            )
    });
    Ok(pending.into())
}

fn finish_report(
    mut collector: crate::replay::TraceCollector,
    sla: SlaThresholds,
) -> TraceSimulationReport {
    collector.set_sla_thresholds(sla);
    collector.finish()
}

fn runtime_error(error: impl std::fmt::Display) -> ReplayError {
    ReplayError::Invariant(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProviderSpec, ReplayAdapters, ReplayRequest, ReplayTopology, WorkerPoolSpec};

    #[test]
    fn replay_spec_lowering_preserves_correlation_routing_and_prompt_provenance() {
        let spec = ReplaySpec {
            version: 1,
            topology: ReplayTopology::Aggregated {
                workers: WorkerPoolSpec::default(),
            },
            engine: serde_json::Value::Null,
            adapters: ReplayAdapters {
                placement: ProviderSpec::round_robin(),
                scaling: ProviderSpec::no_scaling(),
            },
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests: vec![
                ReplayRequest {
                    id: "length-only".into(),
                    arrival_time_ms: 0.0,
                    input_tokens: 3,
                    input_token_ids: None,
                    output_tokens: 2,
                    output_token_ids: None,
                    dp_rank: Some(2),
                    session_id: Some("session-a".into()),
                    turn_index: Some(4),
                    metadata: serde_json::json!({
                        "priority": -7,
                        "strict_priority": 9,
                        "policy_class": "latency",
                        "caller_tag": "preserved"
                    }),
                },
                ReplayRequest {
                    id: "materialized".into(),
                    arrival_time_ms: 1.0,
                    input_tokens: 2,
                    input_token_ids: Some(vec![41, 42]),
                    output_tokens: 1,
                    output_token_ids: None,
                    dp_rank: None,
                    session_id: None,
                    turn_index: None,
                    metadata: serde_json::Value::Null,
                },
            ],
        };

        let lowered = lower_requests(&spec)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        let first = &lowered[0];
        assert_eq!(first.priority, -7);
        assert_eq!(first.strict_priority, 9);
        assert_eq!(first.policy_class.as_deref(), Some("latency"));
        assert_eq!(first.preferred_dp_rank, Some(2));
        assert!(!first.prompt_tokens_are_placement_safe());
        let context = first.replay_context.as_ref().unwrap();
        assert_eq!(context.authored_id, "length-only");
        assert_eq!(context.session_id.as_deref(), Some("session-a"));
        assert_eq!(context.turn_index, Some(4));
        assert_eq!(context.metadata["caller_tag"], "preserved");

        assert_eq!(lowered[1].tokens, vec![41, 42]);
        assert!(lowered[1].prompt_tokens_are_placement_safe());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-owned Router/Planner composition for the shared AISimulate Replayer.

use aisimulate_replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, PoolRoundRobinPlacement,
    ReplayComposition, ReplayScalingPolicy, ReplaySpec, WorkerTopology,
};
use anyhow::{Context, Result, bail};
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
use std::collections::HashSet;

use super::{KvReplayMetadata, KvRouterPlacement};
use crate::common::protocols::MockEngineArgs;
use crate::replay::ReplayPrefillLoadEstimator;
use crate::replay::offline::extensions::kv_events::RouterEventObservation;

/// Round-robin placement with an optional Dynamo Planner scaling policy.
pub(in crate::replay) struct RoundRobinReplayComposition {
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    scaling_enabled: bool,
}

impl RoundRobinReplayComposition {
    pub(in crate::replay) fn new(scaling_policy: Option<Box<dyn ReplayScalingPolicy>>) -> Self {
        Self {
            scaling_enabled: scaling_policy.is_some(),
            scaling_policy,
        }
    }
}

impl ReplayComposition for RoundRobinReplayComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn validate_spec(&self, spec: &ReplaySpec) -> aisimulate_replay::ReplayResult<()> {
        if spec.adapters.placement.provider != "round_robin" {
            return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
                "round-robin composition received placement provider {:?}",
                spec.adapters.placement.provider
            )));
        }
        validate_adapter_descriptors(spec, "round_robin", self.scaling_enabled)?;
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(
        &mut self,
    ) -> aisimulate_replay::ReplayResult<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling_policy.take())
    }

    fn supports_single_worker_runtime(&self) -> bool {
        !self.scaling_enabled
    }
}

enum KvTopologyConfig {
    Aggregated {
        args: Box<MockEngineArgs>,
        num_workers: usize,
    },
    Disaggregated {
        prefill_args: Box<MockEngineArgs>,
        decode_args: Box<MockEngineArgs>,
        num_prefill_workers: usize,
        num_decode_workers: usize,
    },
}

/// KV-aware placement plus an optional Dynamo Planner scaling policy.
pub(in crate::replay) struct KvReplayComposition {
    topology: KvTopologyConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    scaling_enabled: bool,
}

impl KvReplayComposition {
    pub(in crate::replay) fn aggregated(
        args: MockEngineArgs,
        num_workers: usize,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    ) -> Self {
        let scaling_enabled = scaling_policy.is_some();
        Self {
            topology: KvTopologyConfig::Aggregated {
                args: Box::new(args),
                num_workers,
            },
            router_config,
            prefill_load_estimator,
            scaling_policy,
            scaling_enabled,
        }
    }

    pub(in crate::replay) fn disaggregated(
        prefill_args: MockEngineArgs,
        decode_args: MockEngineArgs,
        num_prefill_workers: usize,
        num_decode_workers: usize,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    ) -> Self {
        let scaling_enabled = scaling_policy.is_some();
        Self {
            topology: KvTopologyConfig::Disaggregated {
                prefill_args: Box::new(prefill_args),
                decode_args: Box::new(decode_args),
                num_prefill_workers,
                num_decode_workers,
            },
            router_config,
            prefill_load_estimator,
            scaling_policy,
            scaling_enabled,
        }
    }
}

impl ReplayComposition for KvReplayComposition {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventObservation;
    type AggregatedPlacement = KvRouterPlacement;
    type DisaggregatedPlacement = KvRouterPlacement;

    fn validate_spec(&self, spec: &ReplaySpec) -> aisimulate_replay::ReplayResult<()> {
        if spec.adapters.placement.provider != "dynamo_kv_router" {
            return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
                "Dynamo KV composition received placement provider {:?}",
                spec.adapters.placement.provider
            )));
        }
        validate_adapter_descriptors(spec, "dynamo_kv_router", self.scaling_enabled)?;
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self::AggregatedPlacement> {
        let KvTopologyConfig::Aggregated { args, num_workers } = &self.topology else {
            bail!("disaggregated Router composition used for aggregated replay");
        };
        validate_runtime_topology("aggregated", args, *num_workers, dp_size, &topology)?;
        KvRouterPlacement::new(
            args,
            self.router_config.take(),
            self.prefill_load_estimator.take(),
            topology.len(),
        )
    }

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        let KvTopologyConfig::Disaggregated {
            prefill_args,
            decode_args,
            num_prefill_workers,
            num_decode_workers,
        } = &self.topology
        else {
            bail!("aggregated Router composition used for disaggregated replay");
        };
        validate_runtime_topology(
            "prefill",
            prefill_args,
            *num_prefill_workers,
            prefill_dp_size,
            &prefill_topology,
        )?;
        validate_runtime_topology(
            "decode",
            decode_args,
            *num_decode_workers,
            decode_dp_size,
            &decode_topology,
        )?;
        let router_config = self.router_config.take();
        let prefill = KvRouterPlacement::new(
            prefill_args,
            Some(derive_prefill_router_config(
                prefill_args,
                router_config.clone(),
            )),
            self.prefill_load_estimator.take(),
            prefill_topology.len(),
        )
        .context("constructing prefill KV Router placement")?;
        let decode = KvRouterPlacement::new(
            decode_args,
            Some(derive_decode_router_config(decode_args, router_config)),
            None,
            decode_topology.len(),
        )
        .context("constructing decode KV Router placement")?;
        Ok((prefill, decode))
    }

    fn take_scaling_policy(
        &mut self,
    ) -> aisimulate_replay::ReplayResult<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling_policy.take())
    }
}

fn validate_adapter_descriptors(
    spec: &ReplaySpec,
    expected_placement: &str,
    scaling_enabled: bool,
) -> aisimulate_replay::ReplayResult<()> {
    if spec.adapters.placement.provider != expected_placement {
        return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
            "composition requires placement provider {expected_placement:?}, got {:?}",
            spec.adapters.placement.provider
        )));
    }
    if !spec.adapters.placement.config.is_null() {
        return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
            "placement provider {expected_placement:?} received an unused config descriptor"
        )));
    }

    let expected_scaling = if scaling_enabled {
        "dynamo_planner"
    } else {
        "none"
    };
    if spec.adapters.scaling.provider != expected_scaling {
        return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
            "composition requires scaling provider {expected_scaling:?}, got {:?}",
            spec.adapters.scaling.provider
        )));
    }
    if !spec.adapters.scaling.config.is_null() {
        return Err(aisimulate_replay::ReplayError::InvalidSpec(format!(
            "scaling provider {expected_scaling:?} received an unused config descriptor"
        )));
    }
    Ok(())
}

fn validate_runtime_topology(
    stage: &str,
    args: &MockEngineArgs,
    expected_workers: usize,
    dp_size: u32,
    topology: &[WorkerTopology],
) -> Result<()> {
    let configured_dp = args.dp_size.max(1);
    if dp_size != configured_dp {
        bail!(
            "{stage} Replay topology DP size {dp_size} disagrees with Dynamo engine DP size {configured_dp}"
        );
    }
    if topology.len() != expected_workers {
        bail!(
            "{stage} Replay topology has {} logical workers but Dynamo composition was configured for {expected_workers}",
            topology.len()
        );
    }

    let expected_ranks = usize::try_from(dp_size).context("Replay DP size does not fit usize")?;
    let mut scheduler_ids = HashSet::new();
    for (expected_worker_id, worker) in topology.iter().enumerate() {
        if worker.worker_id != expected_worker_id {
            bail!(
                "{stage} Replay topology worker IDs must be contiguous: expected {expected_worker_id}, got {}",
                worker.worker_id
            );
        }
        if worker.scheduler_ids.len() != expected_ranks {
            bail!(
                "{stage} Replay worker {} exposes {} scheduler ranks; expected DP size {dp_size}",
                worker.worker_id,
                worker.scheduler_ids.len()
            );
        }
        for scheduler_id in &worker.scheduler_ids {
            if !scheduler_ids.insert(*scheduler_id) {
                bail!("{stage} Replay topology repeats scheduler ID {scheduler_id}");
            }
        }
    }
    Ok(())
}

fn base_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
}

pub(in crate::replay) fn derive_prefill_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.router_track_active_blocks = false;
    config
}

pub(in crate::replay) fn derive_decode_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.overlap_score_credit = 0.0;
    config.router_assume_kv_reuse = false;
    config.router_track_prefill_tokens = false;
    config.router_prefill_load_model = RouterPrefillLoadModel::None;
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisimulate_replay::{ProviderSpec, ReplayAdapters, ReplayTopology, WorkerPoolSpec};

    fn spec(scaling: ProviderSpec) -> ReplaySpec {
        ReplaySpec {
            version: 1,
            topology: ReplayTopology::Aggregated {
                workers: WorkerPoolSpec::default(),
            },
            engine: serde_json::Value::Null,
            adapters: ReplayAdapters {
                placement: ProviderSpec::round_robin(),
                scaling,
            },
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: false,
            sla: Default::default(),
            requests: Vec::new(),
        }
    }

    #[test]
    fn injected_scaling_policy_must_match_serializable_descriptor() {
        let composition = RoundRobinReplayComposition::new(None);
        let error = composition
            .validate_spec(&spec(ProviderSpec {
                provider: "dynamo_planner".into(),
                config: serde_json::Value::Null,
            }))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires scaling provider \"none\"")
        );
    }

    #[test]
    fn runtime_topology_must_match_dynamo_dp_and_worker_shape() {
        let args = MockEngineArgs::builder().dp_size(2).build().unwrap();
        let error = validate_runtime_topology(
            "aggregated",
            &args,
            1,
            2,
            &[WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![0],
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("expected DP size 2"));
    }
}

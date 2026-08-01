// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native generalized-engine construction for offline replay.
//!
//! This module contains configuration and conversion helpers only. Scheduler
//! state lives in `aisimulate-engine`; virtual time and worker lifecycle live
//! in the moved aggregated/disaggregated replay runtimes.

use std::num::NonZeroU32;
use std::sync::Arc;

use aisimulate_engine::{
    EngineIdentity, NativeBackend, NativeEngineConfig, NativeEngineFactory,
    NativeGeneralizedEngine, NativeTimingModel, NativeWorkerType,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::{AdmissionQueue, NoReplayMetadata, ReplayEngineObservation, ReplayMode};
use crate::core::EngineEventBatch;
use crate::core::round_robin::PoolRoundRobinPlacement;
use crate::disagg::DisaggRuntimeImpl;
use crate::protocol::DirectRequest;
use crate::replay::OfflineDisaggReplayConfig;
use crate::{ReplayError, ReplayResult, ReplaySpec, Replayer, TraceSimulationReport, WorkerStage};

fn default_dp_size() -> u32 {
    1
}

fn default_tensor_parallel_size() -> u32 {
    1
}

/// Serializable execution-time descriptor for the native AISimulate engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeReplayEngineConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    pub rank: NativeEngineConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<NativeReplayRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<NativeReplayRoleConfig>,
}

impl Default for NativeReplayEngineConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            rank: NativeEngineConfig::default(),
            prefill: None,
            decode: None,
        }
    }
}

/// Rank-group descriptor for one disaggregated role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeReplayRoleConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    pub rank: NativeEngineConfig,
}

impl Default for NativeReplayRoleConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            rank: NativeEngineConfig::default(),
        }
    }
}

impl NativeReplayEngineConfig {
    pub(crate) fn parse(value: &Value) -> ReplayResult<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_value(value.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!("invalid native engine descriptor: {error}"))
        })
    }

    pub(crate) fn role(&self, stage: WorkerStage) -> NativeReplayRoleConfig {
        let mut role = match stage {
            WorkerStage::Aggregated => NativeReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                rank: self.rank.clone(),
            },
            WorkerStage::Prefill => {
                self.prefill
                    .clone()
                    .unwrap_or_else(|| NativeReplayRoleConfig {
                        dp_size: self.dp_size,
                        tensor_parallel_size: self.tensor_parallel_size,
                        rank: self.rank.clone(),
                    })
            }
            WorkerStage::Decode => self
                .decode
                .clone()
                .unwrap_or_else(|| NativeReplayRoleConfig {
                    dp_size: self.dp_size,
                    tensor_parallel_size: self.tensor_parallel_size,
                    rank: self.rank.clone(),
                }),
        };
        role.rank.worker_type = match stage {
            WorkerStage::Aggregated => NativeWorkerType::Aggregated,
            WorkerStage::Prefill => NativeWorkerType::Prefill,
            WorkerStage::Decode => NativeWorkerType::Decode,
        };
        role
    }
}

/// Reusable construction state for one worker role.
#[doc(hidden)]
#[derive(Clone)]
pub struct NativeRoleFactory {
    factory: NativeEngineFactory,
    dp_size: NonZeroU32,
    tensor_parallel_size: u32,
    backend: NativeBackend,
}

impl NativeRoleFactory {
    #[doc(hidden)]
    pub fn build(&self, worker_id: usize) -> ReplayResult<NativeGeneralizedEngine> {
        let worker_id = u64::try_from(worker_id).map_err(|_| {
            ReplayError::Engine(format!(
                "worker id {worker_id} exceeds the native engine range"
            ))
        })?;
        self.factory
            .build(EngineIdentity::new(worker_id), self.dp_size)
            .map_err(engine_error)
    }

    #[doc(hidden)]
    pub fn dp_size(&self) -> u32 {
        self.dp_size.get()
    }

    #[doc(hidden)]
    pub fn gpus_per_worker(&self) -> ReplayResult<usize> {
        usize::try_from(self.dp_size.get())
            .ok()
            .and_then(|dp| {
                usize::try_from(self.tensor_parallel_size)
                    .ok()
                    .and_then(|tp| dp.checked_mul(tp))
            })
            .ok_or_else(|| ReplayError::InvalidSpec("engine GPU count overflows usize".into()))
    }

    #[doc(hidden)]
    pub fn backend(&self) -> NativeBackend {
        self.backend
    }
}

/// Resolves built-in or Runner-provided timing once, then creates role factories.
#[derive(Clone, Default)]
pub struct NativeReplayFactory {
    timing: Option<Arc<dyn NativeTimingModel>>,
    prefill_timing: Option<Arc<dyn NativeTimingModel>>,
    decode_timing: Option<Arc<dyn NativeTimingModel>>,
}

impl NativeReplayFactory {
    pub const fn new() -> Self {
        Self {
            timing: None,
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_timing_model(timing: Arc<dyn NativeTimingModel>) -> Self {
        Self {
            timing: Some(timing),
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_optional_role_timing_models(
        prefill: Option<Arc<dyn NativeTimingModel>>,
        decode: Option<Arc<dyn NativeTimingModel>>,
    ) -> Self {
        Self {
            timing: None,
            prefill_timing: prefill,
            decode_timing: decode,
        }
    }

    #[doc(hidden)]
    pub fn role_factory(
        &self,
        config: &NativeReplayEngineConfig,
        stage: WorkerStage,
        emit_kv_events: bool,
    ) -> ReplayResult<NativeRoleFactory> {
        let mut role = config.role(stage);
        role.rank.emit_kv_events = emit_kv_events;
        let dp_size = NonZeroU32::new(role.dp_size).ok_or_else(|| {
            ReplayError::InvalidSpec("native engine dp_size must be positive".into())
        })?;
        if role.tensor_parallel_size == 0 {
            return Err(ReplayError::InvalidSpec(
                "native tensor_parallel_size must be positive".into(),
            ));
        }
        let timing = match stage {
            WorkerStage::Aggregated => self.timing.as_ref(),
            WorkerStage::Prefill => self.prefill_timing.as_ref().or(self.timing.as_ref()),
            WorkerStage::Decode => self.decode_timing.as_ref().or(self.timing.as_ref()),
        };
        let backend = role.rank.backend;
        let factory = match timing {
            Some(timing) => NativeEngineFactory::with_timing_model(role.rank, Arc::clone(timing)),
            None => NativeEngineFactory::new(role.rank),
        }
        .map_err(engine_error)?;
        Ok(NativeRoleFactory {
            factory,
            dp_size,
            tensor_parallel_size: role.tensor_parallel_size,
            backend,
        })
    }
}

pub fn run_native_replay(spec: ReplaySpec) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(spec, NativeReplayFactory::new())?.run()
}

pub fn run_native_replay_with_timing(
    spec: ReplaySpec,
    timing: Arc<dyn NativeTimingModel>,
) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(spec, NativeReplayFactory::with_timing_model(timing))?.run()
}

pub fn run_native_replay_with_optional_role_timing(
    spec: ReplaySpec,
    prefill: Option<Arc<dyn NativeTimingModel>>,
    decode: Option<Arc<dyn NativeTimingModel>>,
) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(
        spec,
        NativeReplayFactory::with_optional_role_timing_models(prefill, decode),
    )?
    .run()
}

#[derive(Debug, Default)]
struct NativeEventBatch(Vec<aisimulate_engine::NativeKvEvent>);

impl EngineEventBatch for NativeEventBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
struct NativeEventObservation;

impl ReplayEngineObservation for NativeEventObservation {
    type Batch = NativeEventBatch;

    const CAPTURE_NATIVE_KV_EVENTS: bool = true;

    fn observe_native_events(
        _stage: WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        events: Vec<aisimulate_engine::NativeKvEvent>,
    ) -> Self::Batch {
        NativeEventBatch(events)
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|event| match &event.data {
                aisimulate_engine::NativeKvEventData::Stored(stored) => stored.blocks.as_slice(),
                aisimulate_engine::NativeKvEventData::Removed { .. } => &[],
            })
            .map(|block| block.tokens_hash)
            .collect()
    }
}

/// Run the engine-neutral half of Dynamo's live/offline handoff conformance
/// fixture without importing or recompiling Replay implementation sources.
#[doc(hidden)]
pub fn run_native_handoff_conformance(
    config: NativeReplayEngineConfig,
    factory: NativeReplayFactory,
    request: DirectRequest,
) -> ReplayResult<crate::NormalizedHandoffConformance> {
    let prefill_factory = factory.role_factory(&config, WorkerStage::Prefill, true)?;
    let decode_factory = factory.role_factory(&config, WorkerStage::Decode, true)?;
    let backend = prefill_factory.backend();
    if backend != decode_factory.backend() {
        return Err(ReplayError::InvalidSpec(
            "handoff conformance requires matching prefill/decode backends".into(),
        ));
    }
    let runtime_config = OfflineDisaggReplayConfig {
        prefill_factory,
        decode_factory,
        prefill_startup_time_ms: None,
        decode_startup_time_ms: None,
        num_prefill_workers: 1,
        num_decode_workers: 1,
        handoff_latency_ms: 0.0,
    };
    DisaggRuntimeImpl::<
        PoolRoundRobinPlacement<NativeEventBatch>,
        NativeEventObservation,
        NoReplayMetadata,
    >::new_composed(
        &runtime_config,
        AdmissionQueue::new_requests(
            std::collections::VecDeque::from([request]),
            ReplayMode::Trace,
        ),
        true,
        |_, prefill_topology, _, decode_topology| {
            Ok((
                PoolRoundRobinPlacement::new(prefill_topology),
                PoolRoundRobinPlacement::new(decode_topology),
            ))
        },
    )
    .map_err(|error| ReplayError::Invariant(error.to_string()))?
    .run_handoff_conformance(backend)
    .map_err(|error| ReplayError::Invariant(error.to_string()))
}

fn engine_error(error: impl std::fmt::Display) -> ReplayError {
    ReplayError::Engine(error.to_string())
}

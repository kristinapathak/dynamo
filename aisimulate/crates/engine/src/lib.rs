// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral mock inference schedulers and attention-DP composition.

mod cache;
mod common;
mod engine;
pub mod generalized;
mod kv_manager;
mod native;
mod scheduler;
mod trace;

pub use common::running_mean::RunningMean;
pub use common::speculative::normalize_conditional_accept_rates;
pub use engine::{NativeEngineFactory, NativeGeneralizedEngine};
pub use generalized::{
    CommandContext, EngineConfig, EngineEffects, EngineIdentity, EnginePassCompleted,
    EnginePassStarted, GeneralizedMockerEngine, PassId, RankEffects, RankEngine, RankIdentity,
    RankPass, RankPassStarted, SchedulerCommand,
};
pub use native::{
    HandoffId, HandoffTransferTiming, NativeAdmission, NativeBackend, NativeCommand,
    NativeCommandEffects, NativeCommandResult, NativeEngineConfig, NativeForwardPassMetrics,
    NativeKvBlock, NativeKvEvent, NativeKvEventData, NativeLifecycleEvent, NativeMetrics,
    NativeOutput, NativePassCompletionEffects, NativePassStartEffects, NativePreemptionMode,
    NativeRequest, NativeSglangConfig, NativeSglangSchedulePolicy, NativeStoredBlocks,
    NativeTimingModel, NativeTimingModelConfig, NativeTrtllmCapacityPolicy, NativeTrtllmConfig,
    NativeWorkerType, TransferTimingMode, prefill_handoff_delay_ms,
};
pub use scheduler::NativeRankEngine;
#[doc(hidden)]
pub use trace::native_g1_parent_chain_events;

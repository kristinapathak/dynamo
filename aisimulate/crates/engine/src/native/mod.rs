// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral public configuration, command, effect, and timing contracts.

mod config;
mod handoff;
mod protocols;
mod timing;

pub use config::{
    NativeBackend, NativeEngineConfig, NativePreemptionMode, NativeSglangConfig,
    NativeSglangSchedulePolicy, NativeTrtllmCapacityPolicy, NativeTrtllmConfig, NativeWorkerType,
};
pub use handoff::{HandoffId, HandoffTransferTiming, TransferTimingMode, prefill_handoff_delay_ms};
pub use protocols::{
    NativeAdmission, NativeCommand, NativeCommandEffects, NativeCommandResult,
    NativeForwardPassMetrics, NativeKvBlock, NativeKvEvent, NativeKvEventData,
    NativeLifecycleEvent, NativeMetrics, NativeOutput, NativePassCompletionEffects,
    NativePassStartEffects, NativeRequest, NativeStoredBlocks,
};
pub use timing::{NativeTimingModel, NativeTimingModelConfig};

pub(crate) use protocols::NativePendingPass;
pub(crate) use timing::modeled_duration_ms;

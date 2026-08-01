// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Dynamo-to-AISimulate native engine configuration boundary.

use std::sync::Arc;

use aisimulate_engine::{
    NativeBackend, NativeEngineConfig, NativePreemptionMode, NativeSglangConfig,
    NativeSglangSchedulePolicy, NativeTimingModel, NativeTimingModelConfig, NativeWorkerType,
    TransferTimingMode,
};
use aisimulate_replay::{NativeReplayEngineConfig, NativeReplayFactory, NativeReplayRoleConfig};
use anyhow::{Context, Result};
use serde_json::Value;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{
    EngineType, KvTransferTimingMode, MockEngineArgs, PreemptionMode, WorkerType,
};

/// Fully materialized native rank configuration and its process-local timing
/// provider.
pub(crate) struct NativeEngineComponents {
    pub(crate) args: MockEngineArgs,
    pub(crate) rank: NativeEngineConfig,
    pub(crate) timing: Arc<dyn NativeTimingModel>,
}

/// Normalize Dynamo mock-engine arguments and materialize the neutral native
/// rank contract.
///
/// Attention-DP size remains a grouped-engine concern and is intentionally not
/// copied into [`NativeEngineConfig`].
pub(crate) fn native_engine_components(
    args: MockEngineArgs,
    emit_kv_events: bool,
    emit_kv_token_ids: bool,
) -> Result<NativeEngineComponents> {
    let args = args
        .normalized()
        .context("invalid Mocker engine arguments")?;
    let backend = match args.engine_type {
        EngineType::Vllm => NativeBackend::Vllm,
        EngineType::Sglang => NativeBackend::Sglang,
        EngineType::Trtllm => NativeBackend::Trtllm,
    };
    let worker_type = match args.worker_type {
        WorkerType::Aggregated => NativeWorkerType::Aggregated,
        WorkerType::Prefill => NativeWorkerType::Prefill,
        WorkerType::Decode => NativeWorkerType::Decode,
    };
    let preemption_mode = match args.preemption_mode {
        PreemptionMode::Lifo => NativePreemptionMode::Lifo,
        PreemptionMode::Fifo => NativePreemptionMode::Fifo,
    };
    let kv_transfer_timing_mode = match args.kv_transfer_timing_mode {
        KvTransferTimingMode::FullPrompt => TransferTimingMode::FullPrompt,
        KvTransferTimingMode::DestinationMissing => TransferTimingMode::DestinationMissing,
    };
    let sglang_args = args.sglang.as_ref();
    let schedule_policy = match sglang_args.and_then(|sglang| sglang.schedule_policy.as_deref()) {
        Some("lpm") => NativeSglangSchedulePolicy::Lpm,
        Some("fifo") | Some("fcfs") | None => NativeSglangSchedulePolicy::Fifo,
        Some(other) => {
            tracing::warn!(
                schedule_policy = other,
                "unknown SGLang schedule policy; using FIFO"
            );
            NativeSglangSchedulePolicy::Fifo
        }
    };
    let sglang = NativeSglangConfig {
        schedule_policy,
        max_prefill_tokens: sglang_args
            .and_then(|sglang| sglang.max_prefill_tokens)
            .unwrap_or(16_384),
        chunked_prefill_size: sglang_args
            .and_then(|sglang| sglang.chunked_prefill_size)
            .unwrap_or(8_192),
        clip_max_new_tokens: sglang_args
            .and_then(|sglang| sglang.clip_max_new_tokens)
            .unwrap_or(4_096),
        schedule_conservativeness: sglang_args
            .and_then(|sglang| sglang.schedule_conservativeness)
            .unwrap_or(1.0),
    };
    let rank = NativeEngineConfig {
        backend,
        num_gpu_blocks: args.num_gpu_blocks,
        block_size: args.block_size,
        max_model_len: args.max_model_len,
        max_num_seqs: args.max_num_seqs.unwrap_or(usize::MAX),
        max_num_batched_tokens: args.max_num_batched_tokens.unwrap_or(usize::MAX),
        enable_prefix_caching: args.enable_prefix_caching,
        enable_chunked_prefill: args.enable_chunked_prefill,
        speedup_ratio: args.speedup_ratio,
        decode_speedup_ratio: args.decode_speedup_ratio,
        aic_nextn: args.aic_nextn,
        aic_nextn_accept_rates: args.aic_nextn_accept_rates.clone(),
        aic_mtp_seed: args.aic_mtp_seed,
        worker_type,
        preemption_mode,
        emit_kv_events,
        emit_kv_token_ids,
        kv_bytes_per_token: args.kv_bytes_per_token,
        kv_transfer_bandwidth: args.kv_transfer_bandwidth,
        kv_transfer_timing_mode,
        timing_model: NativeTimingModelConfig::External {
            provider: "dynamo_perf_model".to_string(),
            config: Value::Null,
        },
        sglang,
        ..NativeEngineConfig::for_backend(backend)
    };
    let timing: Arc<dyn NativeTimingModel> = Arc::new(DynamoPerfTimingModel {
        inner: Arc::clone(&args.perf_model),
    });
    Ok(NativeEngineComponents { args, rank, timing })
}

fn replay_tensor_parallel_size(args: &MockEngineArgs) -> Result<u32> {
    u32::try_from(args.aic_tp_size.unwrap_or(1))
        .context("Mocker tensor-parallel size exceeds the Replay contract")
}

/// Materialize the serializable engine descriptor and process-local timing
/// provider used by one aggregated Replay invocation.
pub(crate) fn native_aggregated_replay_setup(
    args: &MockEngineArgs,
) -> Result<(NativeReplayEngineConfig, NativeReplayFactory)> {
    let components = native_engine_components(args.clone(), false, false)?;
    let config = NativeReplayEngineConfig {
        dp_size: components.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&components.args)?,
        rank: components.rank,
        prefill: None,
        decode: None,
    };
    Ok((
        config,
        NativeReplayFactory::with_timing_model(components.timing),
    ))
}

/// Materialize role-specific descriptors and timing providers for one
/// disaggregated Replay invocation.
pub(crate) fn native_disaggregated_replay_setup(
    prefill_args: &MockEngineArgs,
    decode_args: &MockEngineArgs,
) -> Result<(NativeReplayEngineConfig, NativeReplayFactory)> {
    let prefill = native_engine_components(prefill_args.clone(), false, false)?;
    let decode = native_engine_components(decode_args.clone(), false, false)?;
    let prefill_role = NativeReplayRoleConfig {
        dp_size: prefill.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&prefill.args)?,
        rank: prefill.rank,
    };
    let decode_role = NativeReplayRoleConfig {
        dp_size: decode.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&decode.args)?,
        rank: decode.rank,
    };
    let config = NativeReplayEngineConfig {
        dp_size: prefill_role.dp_size,
        tensor_parallel_size: prefill_role.tensor_parallel_size,
        rank: prefill_role.rank.clone(),
        prefill: Some(prefill_role),
        decode: Some(decode_role),
    };
    Ok((
        config,
        NativeReplayFactory::with_optional_role_timing_models(
            Some(prefill.timing),
            Some(decode.timing),
        ),
    ))
}

struct DynamoPerfTimingModel {
    inner: Arc<PerfModel>,
}

impl NativeTimingModel for DynamoPerfTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        self.inner
            .predict_prefill_time(batch_size, mean_isl, mean_prefix)
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        self.inner.predict_decode_time(
            batch_size,
            active_kv_tokens,
            mean_context_length,
            total_kv_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::{SglangArgs, TrtllmArgs};

    #[test]
    fn vllm_defaults_materialize_once_at_the_shared_boundary() {
        let args = MockEngineArgs::builder().build().unwrap();
        let components = native_engine_components(args, true, true).unwrap();

        assert_eq!(components.args.block_size, 64);
        assert_eq!(components.rank.backend, NativeBackend::Vllm);
        assert_eq!(components.rank.block_size, 64);
        assert!(components.rank.emit_kv_events);
        assert!(components.rank.emit_kv_token_ids);
    }

    #[test]
    fn backend_specific_fields_match_the_native_contract() {
        let mut sglang = MockEngineArgs::builder().build().unwrap();
        sglang.engine_type = EngineType::Sglang;
        sglang.sglang = Some(SglangArgs {
            schedule_policy: Some("lpm".to_string()),
            page_size: Some(8),
            max_prefill_tokens: Some(512),
            chunked_prefill_size: Some(256),
            clip_max_new_tokens: Some(128),
            schedule_conservativeness: Some(0.5),
        });
        let components = native_engine_components(sglang, false, false).unwrap();
        assert_eq!(components.rank.backend, NativeBackend::Sglang);
        assert_eq!(components.rank.block_size, 8);
        assert_eq!(
            components.rank.sglang.schedule_policy,
            NativeSglangSchedulePolicy::Lpm
        );
        assert_eq!(components.rank.sglang.chunked_prefill_size, 256);

        let mut trtllm = MockEngineArgs::builder().build().unwrap();
        trtllm.engine_type = EngineType::Trtllm;
        trtllm.trtllm = Some(TrtllmArgs::default());
        let components = native_engine_components(trtllm, false, false).unwrap();
        assert_eq!(components.rank.backend, NativeBackend::Trtllm);
        assert_eq!(components.rank.block_size, 32);
    }
}

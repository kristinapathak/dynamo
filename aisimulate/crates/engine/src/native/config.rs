// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serializable native-engine configuration.

use std::sync::Arc;

use anyhow::{Result, ensure};
use serde::{Deserialize, Deserializer, Serialize};

use super::handoff::TransferTimingMode;
use super::timing::{NativeTimingModel, NativeTimingModelConfig, built_in_timing_model};
use crate::common::speculative::normalize_conditional_accept_rates;

const DEFAULT_MAX_PREFILL_TOKENS: usize = 16_384;
const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8_192;
const DEFAULT_CLIP_MAX_NEW_TOKENS: usize = 4_096;
const DEFAULT_SCHEDULE_CONSERVATIVENESS: f64 = 1.0;

fn default_num_gpu_blocks() -> usize {
    16_384
}

fn default_block_size() -> usize {
    64
}

fn default_max_num_seqs() -> usize {
    256
}

fn default_max_num_batched_tokens() -> usize {
    8_192
}

fn default_true() -> bool {
    true
}

fn default_one() -> f64 {
    1.0
}

fn default_aic_mtp_seed() -> u64 {
    42
}

fn default_max_prefill_tokens() -> usize {
    DEFAULT_MAX_PREFILL_TOKENS
}

fn default_chunked_prefill_size() -> usize {
    DEFAULT_CHUNKED_PREFILL_SIZE
}

fn default_clip_max_new_tokens() -> usize {
    DEFAULT_CLIP_MAX_NEW_TOKENS
}

fn default_schedule_conservativeness() -> f64 {
    DEFAULT_SCHEDULE_CONSERVATIVENESS
}

/// Scheduler semantics selected for a native AISimulate rank.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeBackend {
    /// vLLM-style block scheduling.
    #[default]
    Vllm,
    /// SGLang-style radix-cache scheduling.
    Sglang,
    /// TensorRT-LLM scheduling through the shared vLLM-style core.
    Trtllm,
}

impl NativeBackend {
    /// Backend-native KV block size used when a caller does not provide one.
    pub const fn default_block_size(self) -> usize {
        match self {
            Self::Vllm => 64,
            Self::Sglang => 1,
            Self::Trtllm => 32,
        }
    }
}

/// Native scheduler role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeWorkerType {
    /// Prefill and decode execute on the same rank.
    #[default]
    Aggregated,
    /// The rank emits its first token with no separate decode latency.
    Prefill,
    /// The rank performs decode work only.
    Decode,
}

/// Decode preemption victim selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePreemptionMode {
    /// Evict the most recently admitted runnable request.
    #[default]
    Lifo,
    /// Evict the oldest runnable request.
    Fifo,
}

/// SGLang waiting-queue ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSglangSchedulePolicy {
    /// First-in, first-out.
    #[default]
    Fifo,
    /// Longest cached-prefix first for bounded waiting queues.
    Lpm,
}

/// Serializable SGLang scheduler controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeSglangConfig {
    /// Waiting-queue policy.
    pub schedule_policy: NativeSglangSchedulePolicy,
    /// Page-aware prefill-token budget per pass.
    #[serde(default = "default_max_prefill_tokens")]
    pub max_prefill_tokens: usize,
    /// Maximum prompt chunk considered in one pass.
    #[serde(default = "default_chunked_prefill_size")]
    pub chunked_prefill_size: usize,
    /// Output reservation cap used by SGLang admission control.
    #[serde(default = "default_clip_max_new_tokens")]
    pub clip_max_new_tokens: usize,
    /// Multiplier applied to SGLang's adaptive output-reservation ratio.
    #[serde(default = "default_schedule_conservativeness")]
    pub schedule_conservativeness: f64,
}

impl Default for NativeSglangConfig {
    fn default() -> Self {
        Self {
            schedule_policy: NativeSglangSchedulePolicy::Fifo,
            max_prefill_tokens: default_max_prefill_tokens(),
            chunked_prefill_size: default_chunked_prefill_size(),
            clip_max_new_tokens: default_clip_max_new_tokens(),
            schedule_conservativeness: default_schedule_conservativeness(),
        }
    }
}

impl NativeSglangConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_prefill_tokens > 0,
            "sglang.max_prefill_tokens must be positive"
        );
        ensure!(
            self.chunked_prefill_size > 0,
            "sglang.chunked_prefill_size must be positive"
        );
        ensure!(
            self.schedule_conservativeness.is_finite() && self.schedule_conservativeness >= 0.0,
            "sglang.schedule_conservativeness must be finite and non-negative"
        );
        Ok(())
    }
}

/// TensorRT-LLM capacity scheduler policy.
///
/// The native mocker currently models the TensorRT-LLM default only. Keeping
/// the policy explicit prevents a config from silently falling back to vLLM
/// admission or preemption semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTrtllmCapacityPolicy {
    /// Reserve each admitted request through completion and never evict it.
    #[default]
    GuaranteedNoEvict,
}

/// Serializable TensorRT-LLM scheduler controls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeTrtllmConfig {
    /// Capacity scheduler policy.
    pub capacity_scheduler_policy: NativeTrtllmCapacityPolicy,
}

/// Serializable configuration for one native scheduler rank.
///
/// Attention-DP size and worker identity belong to [`crate::EngineConfig`]
/// and [`crate::EngineIdentity`], not this rank-local configuration.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativeEngineConfig {
    /// Scheduler backend whose semantics this rank executes.
    pub backend: NativeBackend,
    /// Physical G1 capacity in blocks.
    #[serde(default = "default_num_gpu_blocks")]
    pub num_gpu_blocks: usize,
    /// KV block size in tokens.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Optional model context limit.
    pub max_model_len: Option<usize>,
    /// Maximum concurrently runnable sequences.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    /// Per-pass token budget.
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
    /// Whether complete blocks remain reusable after request release.
    #[serde(default = "default_true")]
    pub enable_prefix_caching: bool,
    /// Whether a prompt may be split across scheduler passes.
    #[serde(default = "default_true")]
    pub enable_chunked_prefill: bool,
    /// Divisor applied to modeled prefill and decode latency.
    #[serde(default = "default_one")]
    pub speedup_ratio: f64,
    /// Additional divisor applied to decode latency.
    #[serde(default = "default_one")]
    pub decode_speedup_ratio: f64,
    /// MTP/EAGLE draft-token count. One verification forward can emit up to
    /// `aic_nextn + 1` output tokens.
    pub aic_nextn: Option<usize>,
    /// Conditional draft acceptance rates, comma-separated.
    ///
    /// Entry `i` is the probability that draft `i` is accepted given that
    /// every preceding draft was accepted.
    pub aic_nextn_accept_rates: Option<String>,
    /// Base seed for deterministic worker-local MTP acceptance sampling.
    #[serde(default = "default_aic_mtp_seed")]
    pub aic_mtp_seed: u64,
    /// Scheduler role.
    pub worker_type: NativeWorkerType,
    /// Decode preemption victim order.
    pub preemption_mode: NativePreemptionMode,
    /// Retain and expose local token-block hashes in neutral KV events.
    pub emit_kv_events: bool,
    /// Retain block token IDs alongside neutral KV events.
    pub emit_kv_token_ids: bool,
    /// KV-cache bytes occupied by one token for disaggregated transfer timing.
    pub kv_bytes_per_token: Option<usize>,
    /// Modeled prefill-to-decode transfer bandwidth in decimal GB/s.
    pub kv_transfer_bandwidth: Option<f64>,
    /// Prompt footprint used to model disaggregated transfer time.
    pub kv_transfer_timing_mode: TransferTimingMode,
    /// Serializable timing-provider descriptor.
    pub timing_model: NativeTimingModelConfig,
    /// SGLang-only scheduler controls.
    pub sglang: NativeSglangConfig,
    /// TensorRT-LLM-only scheduler controls.
    pub trtllm: NativeTrtllmConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeEngineConfigWire {
    #[serde(default)]
    backend: NativeBackend,
    #[serde(default = "default_num_gpu_blocks")]
    num_gpu_blocks: usize,
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    max_model_len: Option<usize>,
    #[serde(default = "default_max_num_seqs")]
    max_num_seqs: usize,
    #[serde(default = "default_max_num_batched_tokens")]
    max_num_batched_tokens: usize,
    #[serde(default = "default_true")]
    enable_prefix_caching: bool,
    #[serde(default = "default_true")]
    enable_chunked_prefill: bool,
    #[serde(default = "default_one")]
    speedup_ratio: f64,
    #[serde(default = "default_one")]
    decode_speedup_ratio: f64,
    #[serde(default)]
    aic_nextn: Option<usize>,
    #[serde(default)]
    aic_nextn_accept_rates: Option<String>,
    #[serde(default = "default_aic_mtp_seed")]
    aic_mtp_seed: u64,
    #[serde(default)]
    worker_type: NativeWorkerType,
    #[serde(default)]
    preemption_mode: NativePreemptionMode,
    #[serde(default)]
    emit_kv_events: bool,
    #[serde(default)]
    emit_kv_token_ids: bool,
    #[serde(default)]
    kv_bytes_per_token: Option<usize>,
    #[serde(default)]
    kv_transfer_bandwidth: Option<f64>,
    #[serde(default)]
    kv_transfer_timing_mode: TransferTimingMode,
    #[serde(default)]
    timing_model: NativeTimingModelConfig,
    #[serde(default)]
    sglang: NativeSglangConfig,
    #[serde(default)]
    trtllm: NativeTrtllmConfig,
}

impl<'de> Deserialize<'de> for NativeEngineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = NativeEngineConfigWire::deserialize(deserializer)?;
        Ok(Self {
            backend: wire.backend,
            num_gpu_blocks: wire.num_gpu_blocks,
            block_size: wire
                .block_size
                .unwrap_or_else(|| wire.backend.default_block_size()),
            max_model_len: wire.max_model_len,
            max_num_seqs: wire.max_num_seqs,
            max_num_batched_tokens: wire.max_num_batched_tokens,
            enable_prefix_caching: wire.enable_prefix_caching,
            enable_chunked_prefill: wire.enable_chunked_prefill,
            speedup_ratio: wire.speedup_ratio,
            decode_speedup_ratio: wire.decode_speedup_ratio,
            aic_nextn: wire.aic_nextn,
            aic_nextn_accept_rates: wire.aic_nextn_accept_rates,
            aic_mtp_seed: wire.aic_mtp_seed,
            worker_type: wire.worker_type,
            preemption_mode: wire.preemption_mode,
            emit_kv_events: wire.emit_kv_events,
            emit_kv_token_ids: wire.emit_kv_token_ids,
            kv_bytes_per_token: wire.kv_bytes_per_token,
            kv_transfer_bandwidth: wire.kv_transfer_bandwidth,
            kv_transfer_timing_mode: wire.kv_transfer_timing_mode,
            timing_model: wire.timing_model,
            sglang: wire.sglang,
            trtllm: wire.trtllm,
        })
    }
}

impl Default for NativeEngineConfig {
    fn default() -> Self {
        Self {
            backend: NativeBackend::Vllm,
            num_gpu_blocks: default_num_gpu_blocks(),
            block_size: default_block_size(),
            max_model_len: None,
            max_num_seqs: default_max_num_seqs(),
            max_num_batched_tokens: default_max_num_batched_tokens(),
            enable_prefix_caching: true,
            enable_chunked_prefill: true,
            speedup_ratio: 1.0,
            decode_speedup_ratio: 1.0,
            aic_nextn: None,
            aic_nextn_accept_rates: None,
            aic_mtp_seed: default_aic_mtp_seed(),
            worker_type: NativeWorkerType::Aggregated,
            preemption_mode: NativePreemptionMode::Lifo,
            emit_kv_events: false,
            emit_kv_token_ids: false,
            kv_bytes_per_token: None,
            kv_transfer_bandwidth: None,
            kv_transfer_timing_mode: TransferTimingMode::FullPrompt,
            timing_model: NativeTimingModelConfig::Polynomial,
            sglang: NativeSglangConfig::default(),
            trtllm: NativeTrtllmConfig::default(),
        }
    }
}

impl NativeEngineConfig {
    /// Construct backend defaults without inheriting vLLM's block size.
    pub fn for_backend(backend: NativeBackend) -> Self {
        Self {
            backend,
            block_size: backend.default_block_size(),
            ..Self::default()
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.num_gpu_blocks > 0, "num_gpu_blocks must be positive");
        ensure!(self.block_size > 0, "block_size must be positive");
        if matches!(self.backend, NativeBackend::Vllm | NativeBackend::Trtllm) {
            ensure!(
                self.block_size >= 2,
                "vLLM/TRT-LLM block_size must be at least two"
            );
        }
        ensure!(self.max_num_seqs > 0, "max_num_seqs must be positive");
        ensure!(
            self.max_num_batched_tokens > 0,
            "max_num_batched_tokens must be positive"
        );
        ensure!(
            self.max_model_len.is_none_or(|limit| limit > 0),
            "max_model_len must be positive"
        );
        ensure!(
            self.backend == NativeBackend::Vllm || self.max_model_len.is_none(),
            "max_model_len is supported only for backend=vllm"
        );
        ensure!(
            self.speedup_ratio.is_finite() && self.speedup_ratio >= 0.0,
            "speedup_ratio must be finite and non-negative"
        );
        ensure!(
            self.decode_speedup_ratio.is_finite() && self.decode_speedup_ratio >= 0.0,
            "decode_speedup_ratio must be finite and non-negative"
        );
        if let Some(nextn) = self.aic_nextn {
            normalize_conditional_accept_rates(nextn, self.aic_nextn_accept_rates.as_deref())?;
            ensure!(
                self.decode_speedup_ratio == 1.0,
                "aic_nextn requires decode_speedup_ratio=1.0 because MTP output acceleration is modeled by burst sampling"
            );
        } else {
            ensure!(
                self.aic_nextn_accept_rates.is_none(),
                "aic_nextn_accept_rates requires aic_nextn"
            );
        }
        ensure!(
            !self.emit_kv_token_ids || self.emit_kv_events,
            "emit_kv_token_ids requires emit_kv_events"
        );
        ensure!(
            self.kv_bytes_per_token.is_none_or(|bytes| bytes > 0),
            "kv_bytes_per_token must be positive"
        );
        ensure!(
            self.kv_transfer_bandwidth
                .is_none_or(|bandwidth| bandwidth.is_finite() && bandwidth >= 0.0),
            "kv_transfer_bandwidth must be finite and non-negative"
        );
        if self.backend == NativeBackend::Sglang {
            self.sglang.validate()?;
        }
        match &self.timing_model {
            NativeTimingModelConfig::Polynomial => {}
            NativeTimingModelConfig::Fixed {
                prefill_ms,
                decode_ms,
            } => {
                ensure!(
                    prefill_ms.is_finite() && *prefill_ms >= 0.0,
                    "fixed prefill latency must be finite and non-negative"
                );
                ensure!(
                    decode_ms.is_finite() && *decode_ms >= 0.0,
                    "fixed decode latency must be finite and non-negative"
                );
            }
            NativeTimingModelConfig::External { provider, .. } => {
                ensure!(
                    !provider.trim().is_empty(),
                    "timing provider cannot be empty"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn built_in_timing_model(&self) -> Result<Arc<dyn NativeTimingModel>> {
        built_in_timing_model(&self.timing_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialization_uses_backend_native_block_size() {
        for (backend, expected) in [("vllm", 64), ("sglang", 1), ("trtllm", 32)] {
            let config: NativeEngineConfig =
                serde_json::from_value(serde_json::json!({ "backend": backend })).unwrap();
            assert_eq!(config.block_size, expected, "backend={backend}");
        }
    }

    #[test]
    fn deserialization_preserves_an_explicit_block_size() {
        let config: NativeEngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang",
            "block_size": 17
        }))
        .unwrap();
        assert_eq!(config.block_size, 17);
    }

    #[test]
    fn deserialization_still_rejects_unknown_fields() {
        let error = serde_json::from_value::<NativeEngineConfig>(serde_json::json!({
            "backend": "vllm",
            "unknown": true
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn serialization_round_trip_preserves_runtime_neutral_controls() {
        let config = NativeEngineConfig {
            backend: NativeBackend::Sglang,
            block_size: 8,
            num_gpu_blocks: 123,
            max_num_seqs: 7,
            max_num_batched_tokens: 456,
            worker_type: NativeWorkerType::Decode,
            preemption_mode: NativePreemptionMode::Fifo,
            emit_kv_events: true,
            emit_kv_token_ids: true,
            timing_model: NativeTimingModelConfig::Fixed {
                prefill_ms: 2.5,
                decode_ms: 0.75,
            },
            ..NativeEngineConfig::for_backend(NativeBackend::Sglang)
        };
        let encoded = serde_json::to_value(&config).unwrap();
        let decoded: NativeEngineConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn validation_rejects_zero_or_backend_invalid_capacity_fields() {
        let config = NativeEngineConfig {
            num_gpu_blocks: 0,
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("num_gpu_blocks")
        );

        let config = NativeEngineConfig {
            block_size: 1,
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("at least two")
        );

        let config = NativeEngineConfig {
            max_model_len: Some(0),
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_model_len")
        );
    }

    #[test]
    fn validation_accepts_sglang_page_size_one_and_rejects_invalid_controls() {
        let mut config = NativeEngineConfig::for_backend(NativeBackend::Sglang);
        config.validate().unwrap();

        config.sglang.chunked_prefill_size = 0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("chunked_prefill_size")
        );

        let mut config = NativeEngineConfig::for_backend(NativeBackend::Sglang);
        config.sglang.schedule_conservativeness = f64::NAN;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("schedule_conservativeness")
        );
    }

    #[test]
    fn max_model_len_is_vllm_only() {
        for backend in [NativeBackend::Sglang, NativeBackend::Trtllm] {
            let mut config = NativeEngineConfig::for_backend(backend);
            config.max_model_len = Some(128);
            assert!(
                config
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("backend=vllm")
            );
        }
    }

    #[test]
    fn mtp_configuration_validates_rates_and_decode_scaling() {
        let mut config = NativeEngineConfig {
            aic_nextn: Some(2),
            aic_nextn_accept_rates: Some("0.8,0.5".to_string()),
            ..NativeEngineConfig::default()
        };
        config.validate().unwrap();

        config.aic_nextn_accept_rates = Some("1.2".to_string());
        assert!(config.validate().is_err());

        config.aic_nextn_accept_rates = Some("0.8,0.5".to_string());
        config.decode_speedup_ratio = 2.0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("decode_speedup_ratio=1.0")
        );
    }

    #[test]
    fn mtp_rates_require_mtp_to_be_enabled() {
        let config = NativeEngineConfig {
            aic_nextn_accept_rates: Some("0.5".to_string()),
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires aic_nextn")
        );
    }

    #[test]
    fn kv_token_ids_require_kv_event_emission() {
        let config = NativeEngineConfig {
            emit_kv_token_ids: true,
            emit_kv_events: false,
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("emit_kv_token_ids")
        );
    }

    #[test]
    fn timing_provider_descriptors_are_validated_without_loading_them() {
        let config = NativeEngineConfig {
            timing_model: NativeTimingModelConfig::External {
                provider: " ".to_string(),
                config: serde_json::Value::Null,
            },
            ..NativeEngineConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("provider cannot be empty")
        );

        let config = NativeEngineConfig {
            timing_model: NativeTimingModelConfig::Fixed {
                prefill_ms: f64::NAN,
                decode_ms: 1.0,
            },
            ..NativeEngineConfig::default()
        };
        assert!(config.validate().is_err());
    }
}

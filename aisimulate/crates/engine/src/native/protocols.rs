// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Commands and effects exchanged with one native scheduler rank.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{HandoffId, HandoffTransferTiming};

/// Runtime-neutral request accepted by the native rank engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRequest {
    /// Stable request identity allocated by the caller.
    pub request_id: Uuid,
    /// Prompt token IDs.
    pub tokens: Vec<u32>,
    /// Requested output length.
    pub max_output_tokens: usize,
    /// Optional exact output IDs. Its length overrides `max_output_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
}

/// Commands supported by the standalone native scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCommand {
    /// Submit one request.
    Submit(NativeRequest),
    /// Submit a request whose completed prefill KV must remain source-held.
    SubmitHandoffPrefill {
        handoff_id: HandoffId,
        request: NativeRequest,
    },
    /// Accept a decode request and reserve its destination KV footprint.
    ReserveDestination {
        handoff_id: HandoffId,
        request: NativeRequest,
    },
    /// Make a reserved destination runnable after transfer completion.
    ActivateDestination { handoff_id: HandoffId },
    /// Release a successfully transferred source hold.
    ReleaseSource { handoff_id: HandoffId },
    /// Cancel pending or held source ownership.
    CancelSource { handoff_id: HandoffId },
    /// Cancel pending, reserved, or active destination ownership.
    CancelDestination { handoff_id: HandoffId },
    /// Cancel one request.
    ///
    /// A command that removes scheduler-owned state also suppresses output
    /// retained by an in-flight pass. `discard_pending_output` additionally
    /// requests suppression when scheduler cancellation is a no-op, which is
    /// needed after an external driver has already retired the request.
    CancelRequest {
        request_id: Uuid,
        discard_pending_output: bool,
    },
}

/// Result of applying a native scheduler command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCommandResult {
    /// A new request was accepted.
    Submitted(Uuid),
    /// A destination request was accepted; physical reservation may be pending.
    DestinationAccepted { request_id: Uuid },
    /// State or retained effects were changed.
    Applied,
    /// The command addressed no owned state or retained effect.
    Noop,
}

/// Asynchronous scheduler lifecycle fact consumed by the Replayer's handoff
/// coordinator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeLifecycleEvent {
    /// Prefill computation completed and source KV ownership is retained.
    SourceHeld {
        handoff_id: HandoffId,
        request_id: Uuid,
        transfer_timing: HandoffTransferTiming,
    },
    /// Decode-side physical KV capacity has been reserved.
    DestinationReserved {
        handoff_id: HandoffId,
        request_id: Uuid,
        transferable_prompt_tokens: usize,
    },
}

/// One runtime-neutral KV block identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeKvBlock {
    /// Sequence-aware block hash.
    pub block_hash: u64,
    /// Token-only local block hash.
    pub tokens_hash: u64,
    /// Token IDs retained only when explicitly configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
}

/// A consecutive set of newly visible blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeStoredBlocks {
    /// Parent sequence hash immediately preceding the batch.
    pub parent_hash: Option<u64>,
    /// Optional absolute zero-based position of the first block.
    ///
    /// `None` preserves the parent-linked stream emitted by the native
    /// schedulers; adapters must not invent an absolute position because that
    /// changes how downstream radix indexes reconcile stores and removals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_position: Option<usize>,
    /// Blocks in sequence order.
    pub blocks: Vec<NativeKvBlock>,
}

/// Runtime-neutral KV event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeKvEventData {
    /// Blocks became prefix-cache visible.
    Stored(NativeStoredBlocks),
    /// The final physical copy of these hashes was evicted.
    Removed { block_hashes: Vec<u64> },
}

/// Ordered runtime-neutral KV event emitted by one rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeKvEvent {
    /// Monotonic rank-local event sequence.
    pub event_id: u64,
    /// Attention-DP rank.
    pub dp_rank: u32,
    /// Event payload.
    pub data: NativeKvEventData,
}

/// Request admission exposed at pass start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeAdmission {
    /// Admitted request.
    pub request_id: Uuid,
    /// Prompt tokens reused from native G1.
    pub reused_input_tokens: usize,
}

/// One client-visible output.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeOutput {
    /// Request producing the output.
    pub request_id: Uuid,
    /// Generated token, or `None` for a terminal-without-token signal.
    pub token_id: Option<u32>,
    /// Whether request ownership ended with this output.
    pub completed: bool,
    /// Whether admission rejected the request as physically impossible.
    pub rejected: bool,
}

/// Rank-local scheduler and G1 metrics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NativeMetrics {
    pub dp_rank: u32,
    pub active_blocks: u64,
    pub total_blocks: u64,
    pub cache_usage: f64,
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub preemptions_total: u64,
    /// SGLang radix-cache tokens reused by the most recently completed pass.
    ///
    /// Backends without an equivalent pass-local metric report zero.
    pub sglang_cache_hit_tokens: u64,
    /// Total SGLang prefill tokens considered by the most recently completed
    /// pass. Backends without an equivalent pass-local metric report zero.
    pub sglang_cache_total_tokens: u64,
}

/// Per-pass scheduling statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NativeForwardPassMetrics {
    pub num_prefill_requests: u32,
    pub sum_prefill_tokens: u64,
    pub var_prefill_length: f64,
    pub sum_prefill_kv_tokens: u64,
    pub num_decode_requests: u32,
    pub sum_decode_kv_tokens: u64,
    pub var_decode_kv_tokens: f64,
    pub num_queued_prefill: u32,
    pub sum_queued_prefill_tokens: u64,
    pub var_queued_prefill_length: f64,
    pub num_queued_decode: u32,
    pub sum_queued_decode_kv_tokens: u64,
    pub var_queued_decode_kv_tokens: f64,
    pub duration_ms: f64,
}

/// Effects of a scheduler command.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeCommandEffects {
    pub result: NativeCommandResult,
    pub lifecycle_events: Vec<NativeLifecycleEvent>,
    pub kv_events: Vec<NativeKvEvent>,
    /// Requests whose scheduler/KV ownership ended while applying this command.
    ///
    /// Replayers use this authoritative delta for cancellation and handoff
    /// cleanup instead of inferring ownership from command ordering.
    pub retired_requests: Vec<Uuid>,
    /// Scheduler state after the command and all immediately admitted work.
    ///
    /// Live drivers use this to acknowledge commands while the engine is idle
    /// without waiting for an otherwise unrelated forward pass.
    pub metrics: NativeMetrics,
    /// Whether an output already computed by an in-flight pass was suppressed.
    pub suppressed_pending_output: bool,
}

/// Effects visible as soon as a native pass starts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NativePassStartEffects {
    pub admissions: Vec<NativeAdmission>,
    pub kv_events: Vec<NativeKvEvent>,
}

/// Effects released at the modeled pass completion boundary.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NativePassCompletionEffects {
    pub outputs: Vec<NativeOutput>,
    pub lifecycle_events: Vec<NativeLifecycleEvent>,
    /// KV events whose scheduler visibility boundary is pass completion.
    pub kv_events: Vec<NativeKvEvent>,
    pub metrics: NativeMetrics,
    pub forward_pass_metrics: NativeForwardPassMetrics,
}

/// Retained completion effects of an eagerly executed native pass.
#[doc(hidden)]
pub struct NativePendingPass {
    pub(crate) effects: NativePassCompletionEffects,
}

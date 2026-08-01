// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin neutral contract adapter over the mechanically moved scheduler cores.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{
    DirectRequest, EngineType, KvTransferTimingMode, MockEngineArgs, PreemptionMode, SglangArgs,
    WorkerType,
};
use crate::generalized::{CommandContext, RankEngine, RankIdentity, RankPass};
use crate::native::{
    HandoffId, NativeAdmission, NativeBackend, NativeCommand, NativeCommandEffects,
    NativeCommandResult, NativeEngineConfig, NativeForwardPassMetrics, NativeLifecycleEvent,
    NativeMetrics, NativeOutput, NativePassCompletionEffects, NativePassStartEffects,
    NativePendingPass, NativePreemptionMode, NativeRequest, NativeTimingModel, NativeWorkerType,
    TransferTimingMode,
};

use super::{
    EngineCore, EnginePassResult, KvEventVisibility, MockerMetrics,
    SchedulerCommand as CoreCommand, SchedulerCommandEffects as CoreCommandEffects,
    SchedulerCommandResult as CoreCommandResult, SchedulerLifecycleEvent as CoreLifecycle,
    SglangCore, VllmCore,
};

pub fn native_seed_offset(identity: RankIdentity) -> Result<u64> {
    identity
        .worker_id
        .checked_mul(u64::from(identity.dp_size.get()))
        .and_then(|base| base.checked_add(u64::from(identity.dp_rank)))
        .ok_or_else(|| anyhow!("native mock-engine seed offset overflow"))
}

/// One preserved vLLM/SGLang scheduler rank behind the neutral contract.
pub struct NativeRankEngine {
    core: EngineCore,
    handoff_requests: HashMap<HandoffId, Uuid>,
}

impl NativeRankEngine {
    pub fn new_with_timing_model(
        identity: RankIdentity,
        config: &NativeEngineConfig,
        timing: Arc<dyn NativeTimingModel>,
        seed_offset: u64,
    ) -> Result<Self> {
        config.validate()?;
        let args = core_args(identity, config, timing);
        let capture_kv_events = config.emit_kv_events;
        let core = match config.backend {
            NativeBackend::Vllm | NativeBackend::Trtllm => {
                EngineCore::Vllm(VllmCore::new_with_worker_rank(
                    args,
                    identity.worker_id,
                    identity.dp_rank,
                    seed_offset,
                    capture_kv_events,
                ))
            }
            NativeBackend::Sglang => EngineCore::Sglang(SglangCore::new_with_worker_rank(
                args,
                identity.worker_id,
                identity.dp_rank,
                seed_offset,
                capture_kv_events,
            )),
        };
        Ok(Self {
            core,
            handoff_requests: HashMap::new(),
        })
    }

    fn core_command(command: NativeCommand) -> CoreCommand {
        match command {
            NativeCommand::Submit(request) => CoreCommand::Submit(core_request(request)),
            NativeCommand::CancelRequest { request_id, .. } => {
                CoreCommand::CancelRequest { request_id }
            }
            NativeCommand::SubmitHandoffPrefill {
                handoff_id,
                request,
            } => CoreCommand::SubmitHandoffPrefill {
                handoff_id,
                request: core_request(request),
            },
            NativeCommand::ReserveDestination {
                handoff_id,
                request,
            } => CoreCommand::ReserveDestination {
                handoff_id,
                request: core_request(request),
            },
            NativeCommand::ActivateDestination { handoff_id } => {
                CoreCommand::ActivateDestination { handoff_id }
            }
            NativeCommand::ReleaseSource { handoff_id } => {
                CoreCommand::ReleaseSource { handoff_id }
            }
            NativeCommand::CancelSource { handoff_id } => CoreCommand::CancelSource { handoff_id },
            NativeCommand::CancelDestination { handoff_id } => {
                CoreCommand::CancelDestination { handoff_id }
            }
        }
    }

    fn metrics(&self) -> NativeMetrics {
        let metrics = match &self.core {
            EngineCore::Vllm(core) => core.mocker_metrics(),
            EngineCore::Sglang(core) => core.mocker_metrics(),
        };
        native_metrics(metrics)
    }
}

impl RankEngine for NativeRankEngine {
    type Config = NativeEngineConfig;
    type Command = NativeCommand;
    type CommandEffects = NativeCommandEffects;
    type PassStartEffects = NativePassStartEffects;
    type PendingPass = NativePendingPass;
    type PassCompletionEffects = NativePassCompletionEffects;
    type InternalEffects = ();

    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        let timing = config.built_in_timing_model()?;
        let seed_offset = native_seed_offset(identity)?;
        Self::new_with_timing_model(identity, config, timing, seed_offset)
    }

    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        let pending_suppression = pending_output_suppression(&command, &self.handoff_requests);
        let handoff_update = handoff_tracking_update(&command);
        let core_command = Self::core_command(command);
        let mut effects = self
            .core
            .apply_command_effects(core_command, context.allow_immediate_admission())?;
        // Preserve the scheduler's command boundary: native G1 publishes KV
        // mutations into the rank-local capture sink, while command effects
        // are the only observation returned to the driver at this point. If
        // these events remain buffered, a later pass may drain and discard an
        // incomplete prefix of the stream before the Router observes it.
        effects.kv_events.extend(self.core.drain_kv_events());
        let suppressed_pending_output = if let (Some((request_id, discard_on_noop)), Some(pending)) =
            (pending_suppression, pending_pass)
            && (effects.result != CoreCommandResult::Noop || discard_on_noop)
        {
            let before = pending.effects.outputs.len();
            pending
                .effects
                .outputs
                .retain(|output| output.request_id != request_id);
            pending
                .effects
                .lifecycle_events
                .retain(|event| match *event {
                    NativeLifecycleEvent::SourceHeld { request_id: id, .. }
                    | NativeLifecycleEvent::DestinationReserved { request_id: id, .. } => {
                        id != request_id
                    }
                });
            before != pending.effects.outputs.len()
        } else {
            false
        };
        if effects.result != CoreCommandResult::Noop || suppressed_pending_output {
            self.apply_handoff_tracking_update(handoff_update);
        }
        for request_id in &effects.retired_requests {
            self.handoff_requests
                .retain(|_, tracked_request| tracked_request != request_id);
        }
        native_command_effects(effects, self.metrics(), suppressed_pending_output)
    }

    fn is_ready(&self) -> bool {
        !self.core.is_drained()
    }

    fn waiting_for_external_command(&self) -> bool {
        self.core.waiting_for_external_command()
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        let pass = self.core.try_execute_hidden_pass(now_ms)?;
        let end_ms = pass.end_ms;
        let (start_effects, completion_effects) = split_pass(pass)?;
        Ok(RankPass {
            end_ms,
            start_effects,
            pending: NativePendingPass {
                effects: completion_effects,
            },
        })
    }

    fn complete_pass(
        &mut self,
        mut pending: Self::PendingPass,
        _end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        // The preserved scheduler retries deferred destination reservations
        // when a forward pass releases capacity. Keep that wakeup at the
        // pass-completion boundary: command-time retry is suppressed while a
        // pass is in flight, and waiting until an unrelated later pass can
        // leave disaggregated replay permanently asleep.
        pending.effects.lifecycle_events.extend(
            self.core
                .retry_pending_destinations()
                .into_iter()
                .map(native_lifecycle),
        );
        let completion_kv_events = self.core.drain_kv_events();
        pending.effects.kv_events.extend(completion_kv_events);
        pending.effects.metrics = self.metrics();
        for output in &pending.effects.outputs {
            if output.completed {
                self.handoff_requests
                    .retain(|_, request_id| *request_id != output.request_id);
            }
        }
        Ok(pending.effects)
    }

    fn complete_idle_group_pass(
        &mut self,
        started_at_ms: f64,
        end_ms: f64,
    ) -> Result<Option<Self::PassCompletionEffects>> {
        let lifecycle_events = self
            .core
            .retry_pending_destinations()
            .into_iter()
            .map(native_lifecycle)
            .collect::<Vec<_>>();
        let kv_events = self.core.drain_kv_events();
        Ok(Some(NativePassCompletionEffects {
            lifecycle_events,
            kv_events,
            metrics: self.metrics(),
            forward_pass_metrics: NativeForwardPassMetrics {
                duration_ms: (end_ms - started_at_ms).max(0.0),
                ..Default::default()
            },
            ..NativePassCompletionEffects::default()
        }))
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        None
    }

    fn process_internal_work(
        &mut self,
        _now_ms: f64,
        _pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.core.is_drained()
    }
}

#[derive(Clone, Copy)]
enum HandoffTrackingUpdate {
    None,
    Insert(HandoffId, Uuid),
    RemoveHandoff(HandoffId),
    RemoveRequest(Uuid),
}

impl NativeRankEngine {
    fn apply_handoff_tracking_update(&mut self, update: HandoffTrackingUpdate) {
        match update {
            HandoffTrackingUpdate::None => {}
            HandoffTrackingUpdate::Insert(handoff_id, request_id) => {
                self.handoff_requests.insert(handoff_id, request_id);
            }
            HandoffTrackingUpdate::RemoveHandoff(handoff_id) => {
                self.handoff_requests.remove(&handoff_id);
            }
            HandoffTrackingUpdate::RemoveRequest(request_id) => self
                .handoff_requests
                .retain(|_, tracked_request| *tracked_request != request_id),
        }
    }
}

fn handoff_tracking_update(command: &NativeCommand) -> HandoffTrackingUpdate {
    match command {
        NativeCommand::SubmitHandoffPrefill {
            handoff_id,
            request,
        }
        | NativeCommand::ReserveDestination {
            handoff_id,
            request,
        } => HandoffTrackingUpdate::Insert(*handoff_id, request.request_id),
        NativeCommand::ReleaseSource { handoff_id }
        | NativeCommand::CancelSource { handoff_id }
        | NativeCommand::CancelDestination { handoff_id } => {
            HandoffTrackingUpdate::RemoveHandoff(*handoff_id)
        }
        NativeCommand::CancelRequest { request_id, .. } => {
            HandoffTrackingUpdate::RemoveRequest(*request_id)
        }
        NativeCommand::Submit(_) | NativeCommand::ActivateDestination { .. } => {
            HandoffTrackingUpdate::None
        }
    }
}

fn core_args(
    _identity: RankIdentity,
    config: &NativeEngineConfig,
    timing: Arc<dyn NativeTimingModel>,
) -> MockEngineArgs {
    MockEngineArgs {
        engine_type: match config.backend {
            NativeBackend::Vllm => EngineType::Vllm,
            NativeBackend::Sglang => EngineType::Sglang,
            NativeBackend::Trtllm => EngineType::Trtllm,
        },
        num_gpu_blocks: config.num_gpu_blocks,
        block_size: config.block_size,
        max_model_len: config.max_model_len,
        max_num_seqs: Some(config.max_num_seqs),
        max_num_batched_tokens: Some(config.max_num_batched_tokens),
        enable_prefix_caching: config.enable_prefix_caching,
        enable_chunked_prefill: config.enable_chunked_prefill,
        speedup_ratio: config.speedup_ratio,
        decode_speedup_ratio: config.decode_speedup_ratio,
        worker_type: match config.worker_type {
            NativeWorkerType::Aggregated => WorkerType::Aggregated,
            NativeWorkerType::Prefill => WorkerType::Prefill,
            NativeWorkerType::Decode => WorkerType::Decode,
        },
        perf_model: Arc::new(PerfModel::External { timing }),
        aic_nextn: config.aic_nextn,
        aic_nextn_accept_rates: config.aic_nextn_accept_rates.clone(),
        aic_mtp_seed: config.aic_mtp_seed,
        kv_bytes_per_token: config.kv_bytes_per_token,
        kv_transfer_bandwidth: config.kv_transfer_bandwidth,
        kv_transfer_timing_mode: match config.kv_transfer_timing_mode {
            TransferTimingMode::FullPrompt => KvTransferTimingMode::FullPrompt,
            TransferTimingMode::DestinationMissing => KvTransferTimingMode::DestinationMissing,
        },
        preemption_mode: match config.preemption_mode {
            NativePreemptionMode::Lifo => PreemptionMode::Lifo,
            NativePreemptionMode::Fifo => PreemptionMode::Fifo,
        },
        sglang: Some(SglangArgs {
            schedule_policy: Some(
                match config.sglang.schedule_policy {
                    crate::NativeSglangSchedulePolicy::Fifo => "fifo",
                    crate::NativeSglangSchedulePolicy::Lpm => "lpm",
                }
                .to_string(),
            ),
            page_size: Some(config.block_size),
            max_prefill_tokens: Some(config.sglang.max_prefill_tokens),
            chunked_prefill_size: Some(config.sglang.chunked_prefill_size),
            clip_max_new_tokens: Some(config.sglang.clip_max_new_tokens),
            schedule_conservativeness: Some(config.sglang.schedule_conservativeness),
        }),
        emit_kv_events: config.emit_kv_events,
        emit_kv_token_ids: config.emit_kv_token_ids,
    }
}

fn core_request(request: NativeRequest) -> DirectRequest {
    DirectRequest {
        tokens: request.tokens,
        max_output_tokens: request.max_output_tokens,
        output_token_ids: request.output_token_ids,
        uuid: Some(request.request_id),
        dp_rank: 0,
        arrival_timestamp_ms: None,
    }
}

fn pending_output_suppression(
    command: &NativeCommand,
    handoffs: &HashMap<HandoffId, Uuid>,
) -> Option<(Uuid, bool)> {
    match *command {
        NativeCommand::CancelRequest {
            request_id,
            discard_pending_output,
        } => Some((request_id, discard_pending_output)),
        NativeCommand::CancelSource { handoff_id }
        | NativeCommand::CancelDestination { handoff_id } => handoffs
            .get(&handoff_id)
            .copied()
            .map(|request_id| (request_id, false)),
        _ => None,
    }
}

fn native_command_effects(
    effects: CoreCommandEffects,
    metrics: NativeMetrics,
    suppressed_pending_output: bool,
) -> Result<NativeCommandEffects> {
    let result = match effects.result {
        CoreCommandResult::Submitted(id) => NativeCommandResult::Submitted(id),
        CoreCommandResult::DestinationAccepted { request_id } => {
            NativeCommandResult::DestinationAccepted { request_id }
        }
        CoreCommandResult::Applied => NativeCommandResult::Applied,
        CoreCommandResult::Noop => NativeCommandResult::Noop,
    };
    Ok(NativeCommandEffects {
        result,
        lifecycle_events: effects
            .lifecycle_events
            .into_iter()
            .map(native_lifecycle)
            .collect(),
        kv_events: effects.kv_events,
        retired_requests: effects.retired_requests,
        metrics,
        suppressed_pending_output,
    })
}

fn native_lifecycle(event: CoreLifecycle) -> NativeLifecycleEvent {
    match event {
        CoreLifecycle::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        } => NativeLifecycleEvent::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        },
        CoreLifecycle::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        } => NativeLifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        },
    }
}

fn native_metrics(metrics: MockerMetrics) -> NativeMetrics {
    NativeMetrics {
        dp_rank: metrics.dp_rank,
        active_blocks: metrics.active_decode_blocks,
        total_blocks: metrics.total_blocks,
        cache_usage: metrics.gpu_cache_usage_perc,
        running_requests: metrics.running_requests,
        waiting_requests: metrics.waiting_requests,
        preemptions_total: metrics.vllm_preemptions_total,
        sglang_cache_hit_tokens: metrics.sglang_cache_hit_tokens,
        sglang_cache_total_tokens: metrics.sglang_cache_total_tokens,
    }
}

fn split_pass(
    pass: EnginePassResult,
) -> Result<(NativePassStartEffects, NativePassCompletionEffects)> {
    let EnginePassResult {
        output_signals,
        admissions,
        lifecycle_events,
        mocker_metrics,
        kv_event_visibility,
        kv_events,
        fpm,
        ..
    } = pass;
    let (start_kv, completion_kv) = match kv_event_visibility {
        KvEventVisibility::PassStart => (kv_events, Vec::new()),
        KvEventVisibility::PassEnd => (Vec::new(), kv_events),
    };
    let start = NativePassStartEffects {
        admissions: admissions
            .into_iter()
            .map(|admission| NativeAdmission {
                request_id: admission.uuid,
                reused_input_tokens: admission.reused_input_tokens,
            })
            .collect(),
        kv_events: start_kv,
    };
    let completion = NativePassCompletionEffects {
        outputs: output_signals
            .into_iter()
            .map(|output| NativeOutput {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
            })
            .collect(),
        lifecycle_events: lifecycle_events.into_iter().map(native_lifecycle).collect(),
        kv_events: completion_kv,
        metrics: native_metrics(mocker_metrics),
        forward_pass_metrics: fpm.map(native_fpm).unwrap_or_default(),
    };
    Ok((start, completion))
}

fn native_fpm(fpm: crate::common::protocols::ForwardPassSnapshot) -> NativeForwardPassMetrics {
    NativeForwardPassMetrics {
        num_prefill_requests: fpm.num_prefill_requests,
        sum_prefill_tokens: fpm.sum_prefill_tokens,
        var_prefill_length: fpm.var_prefill_length,
        sum_prefill_kv_tokens: fpm.sum_prefill_kv_tokens,
        num_decode_requests: fpm.num_decode_requests,
        sum_decode_kv_tokens: fpm.sum_decode_kv_tokens,
        var_decode_kv_tokens: fpm.var_decode_kv_tokens,
        num_queued_prefill: fpm.num_queued_prefill,
        sum_queued_prefill_tokens: fpm.sum_queued_prefill_tokens,
        var_queued_prefill_length: fpm.var_queued_prefill_length,
        num_queued_decode: fpm.num_queued_decode,
        sum_queued_decode_kv_tokens: fpm.sum_queued_decode_kv_tokens,
        var_queued_decode_kv_tokens: fpm.var_queued_decode_kv_tokens,
        duration_ms: fpm.wall_time_secs * 1_000.0,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::native::NativeTimingModelConfig;

    fn rank() -> NativeRankEngine {
        rank_for_worker(NativeWorkerType::Aggregated)
    }

    fn rank_for_worker(worker_type: NativeWorkerType) -> NativeRankEngine {
        let config = NativeEngineConfig {
            worker_type,
            num_gpu_blocks: 8,
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 16,
            speedup_ratio: 0.0,
            timing_model: NativeTimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..NativeEngineConfig::default()
        };
        NativeRankEngine::new(
            RankIdentity {
                worker_id: 1,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap()
    }

    fn start_request_pass(
        rank: &mut NativeRankEngine,
        request_id: Uuid,
        output_token_ids: Vec<u32>,
    ) -> NativePendingPass {
        let effects = rank
            .apply_command_effects(
                NativeCommand::Submit(NativeRequest {
                    request_id,
                    tokens: vec![1, 2, 3, 4],
                    max_output_tokens: output_token_ids.len(),
                    output_token_ids: Some(output_token_ids),
                }),
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, NativeCommandResult::Submitted(request_id));
        let pass = rank.execute_pass(0.0).unwrap();
        assert!(
            pass.pending
                .effects
                .outputs
                .iter()
                .any(|output| output.request_id == request_id)
        );
        pass.pending
    }

    #[test]
    fn ordinary_cancel_suppresses_pending_output_when_scheduler_state_is_removed() {
        let request_id = Uuid::from_u128(90_001);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5, 6]);

        let effects = rank
            .apply_command_effects(
                NativeCommand::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, NativeCommandResult::Applied);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn ordinary_noop_cancel_preserves_pending_output() {
        let request_id = Uuid::from_u128(90_002);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                NativeCommand::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, NativeCommandResult::Noop);
        assert!(!effects.suppressed_pending_output);
        assert_eq!(pending.effects.outputs.len(), 1);
    }

    #[test]
    fn explicit_discard_suppresses_pending_output_after_noop_cancellation() {
        let request_id = Uuid::from_u128(90_003);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                NativeCommand::CancelRequest {
                    request_id,
                    discard_pending_output: true,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, NativeCommandResult::Noop);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn handoff_tracking_is_inserted_on_success_and_cleared_by_cancel() {
        let mut rank = rank_for_worker(NativeWorkerType::Decode);
        let handoff_id = HandoffId::from(Uuid::from_u128(91_001));
        let request_id = Uuid::from_u128(91_002);
        let reservation = rank
            .apply_command_effects(
                NativeCommand::ReserveDestination {
                    handoff_id,
                    request: NativeRequest {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert!(matches!(
            reservation.result,
            NativeCommandResult::DestinationAccepted { .. }
        ));
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let cancellation = rank
            .apply_command_effects(
                NativeCommand::CancelDestination { handoff_id },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(cancellation.result, NativeCommandResult::Applied);
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }

    #[test]
    fn terminal_handoff_output_clears_tracking() {
        let mut rank = rank_for_worker(NativeWorkerType::Prefill);
        let handoff_id = HandoffId::from(Uuid::from_u128(92_001));
        let request_id = Uuid::from_u128(92_002);
        let submission = rank
            .apply_command_effects(
                NativeCommand::SubmitHandoffPrefill {
                    handoff_id,
                    request: NativeRequest {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(
            submission.result,
            NativeCommandResult::Submitted(request_id)
        );
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let pass = rank.execute_pass(0.0).unwrap();
        let completion = rank.complete_pass(pass.pending, pass.end_ms).unwrap();
        assert!(
            completion
                .outputs
                .iter()
                .any(|output| output.request_id == request_id && output.completed)
        );
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }
}

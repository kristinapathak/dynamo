// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_engine::{
    NativeCommand, NativeCommandResult, NativeGeneralizedEngine, NativeKvEvent, NativeRequest,
    SchedulerCommand,
};
use anyhow::{Context, Result, bail};
use uuid::Uuid;

use super::AdmissionEvent;
use crate::native::NativeRoleFactory;
use crate::protocol::{DirectRequest, OutputSignal};
use crate::replay::TraceCollector;

pub(crate) struct ReplayWorkerPass {
    pub(crate) end_ms: f64,
    pub(crate) completed_requests: usize,
    pub(crate) output_signals: Vec<OutputSignal>,
    pub(crate) admissions: Vec<AdmissionEvent>,
    /// KV observations emitted while the pass is scheduled. Backends with
    /// pass-start publication semantics populate this batch.
    pub(crate) pass_start_kv_events: Vec<NativeKvEvent>,
    /// KV observations emitted when the pass completes. Backends with
    /// pass-end publication semantics populate this batch.
    pub(crate) pass_end_kv_events: Vec<NativeKvEvent>,
}

impl ReplayWorkerPass {
    pub(crate) fn has_kv_events(&self) -> bool {
        !self.pass_start_kv_events.is_empty() || !self.pass_end_kv_events.is_empty()
    }
}

/// Synchronous DP1 driver used by the preserved single-worker replay loop.
pub(crate) struct ReplayWorkerCore {
    engine: NativeGeneralizedEngine,
    in_flight: usize,
}

impl ReplayWorkerCore {
    pub(crate) fn new(factory: NativeRoleFactory) -> Result<Self> {
        Ok(Self {
            engine: factory.build(0)?,
            in_flight: 0,
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.in_flight == 0 && self.engine.is_drained()
    }

    pub(crate) fn receive(&mut self, request: DirectRequest, now_ms: f64) -> Result<Uuid> {
        let request_id = request
            .uuid
            .context("single-worker replay request must have a UUID")?;
        let effects = self
            .engine
            .apply_command_effects(
                SchedulerCommand::new(
                    0,
                    NativeCommand::Submit(NativeRequest {
                        request_id,
                        tokens: request.tokens,
                        max_output_tokens: request.max_output_tokens,
                        output_token_ids: request.output_token_ids,
                    }),
                ),
                now_ms,
            )?
            .into_by_rank()
            .into_iter()
            .next()
            .context("single-rank submit produced no command effects")?
            .effects;
        if !matches!(effects.result, NativeCommandResult::Submitted(id) if id == request_id) {
            bail!("single-worker native engine returned an unexpected submit result");
        }
        self.in_flight = self
            .in_flight
            .checked_add(1)
            .context("single-worker in-flight request count overflow")?
            .checked_sub(effects.retired_requests.len())
            .context("single-worker native engine retired an untracked request")?;
        Ok(request_id)
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.in_flight
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> Result<ReplayWorkerPass> {
        let started = self
            .engine
            .execute_pass(now_ms)?
            .context("single-worker replay attempted a pass with no ready work")?;
        let mut admissions = Vec::new();
        let mut pass_start_kv_events = Vec::new();
        for rank in started.by_rank {
            debug_assert_eq!(rank.dp_rank, 0);
            for admission in rank.effects.admissions {
                collector.on_admit(admission.request_id, now_ms, admission.reused_input_tokens);
                admissions.push(AdmissionEvent {
                    uuid: admission.request_id,
                    reused_input_tokens: admission.reused_input_tokens,
                });
            }
            pass_start_kv_events.extend(rank.effects.kv_events);
        }

        let completed = self.engine.complete_pass(started.pass_id, started.end_ms)?;
        let mut output_signals = Vec::new();
        let mut pass_end_kv_events = Vec::new();
        for rank in completed.effects.into_by_rank() {
            debug_assert_eq!(rank.dp_rank, 0);
            pass_end_kv_events.extend(rank.effects.kv_events);
            for output in rank.effects.outputs {
                if output.token_id.is_some() {
                    collector.on_token(output.request_id, started.end_ms);
                }
                output_signals.push(OutputSignal {
                    uuid: output.request_id,
                    token_id: output.token_id,
                    completed: output.completed,
                    rejected: output.rejected,
                    handoff_delay_ms: None,
                });
            }
        }
        let completed_requests = output_signals
            .iter()
            .filter(|signal| signal.completed)
            .count();
        self.in_flight = self
            .in_flight
            .checked_sub(completed_requests)
            .context("single-worker completed more requests than replay tracked")?;
        Ok(ReplayWorkerPass {
            end_ms: started.end_ms,
            completed_requests,
            output_signals,
            admissions,
            pass_start_kv_events,
            pass_end_kv_events,
        })
    }
}

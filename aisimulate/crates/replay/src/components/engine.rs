// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::marker::PhantomData;

use aisimulate_engine::{
    NativeCommand, NativeCommandResult, NativeForwardPassMetrics, NativeGeneralizedEngine,
    NativePassCompletionEffects, NativeRequest, PassId, SchedulerCommand,
};
use anyhow::{Context, Result, bail};
use uuid::Uuid;

use super::super::core::{EngineEventBatch, EngineProgress, NoEngineEvents, WorkerTopology};
use super::super::events::{EnginePassCompletion, SimulationWorkerStage, WorkerCompletionPayload};
use super::{EngineEffects, EnginePassMode, ObservedCommandEffects, ReplayEngineObservation};
use crate::native::NativeRoleFactory;
use crate::protocol::{DirectRequest, ForwardPassSnapshot, OutputSignal};
use crate::replay::TraceCollector;

#[derive(Clone, Copy)]
struct PendingPass {
    pass_id: PassId,
    started_at_ms: f64,
    end_ms: f64,
}

struct LogicalWorker {
    engine: NativeGeneralizedEngine,
    scheduler_ids: Vec<usize>,
    /// Request IDs whose ownership is still counted by Replay for each rank.
    ///
    /// A handoff prefill can emit a terminal source signal before the later
    /// `ReleaseSource` command retires the engine-side hold. Tracking IDs, not
    /// only a counter, makes the second retirement idempotent while still
    /// accounting for cancellation before prefill completion.
    in_flight_by_rank: Vec<BTreeSet<Uuid>>,
    pending_pass: Option<PendingPass>,
}

impl LogicalWorker {
    fn total_in_flight(&self) -> usize {
        self.in_flight_by_rank.iter().map(BTreeSet::len).sum()
    }
}

#[derive(Clone, Copy)]
struct SchedulerOwner {
    worker_id: usize,
    dp_rank: u32,
}

/// Fleet/lifecycle adapter around the authoritative generalized mock engine.
///
/// This type owns stable scheduler IDs, startup/draining state, and replay
/// accounting. Attention-DP readiness, pass barriers, and group completion
/// time are owned exclusively by [`NativeGeneralizedEngine`].
pub(crate) struct EngineComponent<Observation = NoEngineEvents>
where
    Observation: ReplayEngineObservation,
{
    stage: SimulationWorkerStage,
    _pass_mode: EnginePassMode,
    workers: Vec<Option<LogicalWorker>>,
    scheduler_owners: Vec<Option<SchedulerOwner>>,
    live_worker_count: usize,
    total_in_flight: usize,
    ready_workers: BTreeSet<usize>,
    pending_removal: BTreeSet<usize>,
    pending_startup: BTreeSet<usize>,
    factory: NativeRoleFactory,
    startup_time_ms: Option<f64>,
    observation: PhantomData<Observation>,
}

impl<Observation> EngineComponent<Observation>
where
    Observation: ReplayEngineObservation,
{
    pub(crate) fn new_with_factory(
        stage: SimulationWorkerStage,
        pass_mode: EnginePassMode,
        factory: NativeRoleFactory,
        num_workers: usize,
        startup_time_ms: Option<f64>,
    ) -> Result<Self> {
        let mut component = Self {
            stage,
            _pass_mode: pass_mode,
            workers: Vec::with_capacity(num_workers),
            scheduler_owners: Vec::with_capacity(
                num_workers.saturating_mul(factory.dp_size() as usize),
            ),
            live_worker_count: 0,
            total_in_flight: 0,
            ready_workers: BTreeSet::new(),
            pending_removal: BTreeSet::new(),
            pending_startup: BTreeSet::new(),
            factory,
            startup_time_ms,
            observation: PhantomData,
        };
        for _ in 0..num_workers {
            component.add_worker()?;
        }
        component.refresh_all_workers();
        Ok(component)
    }

    fn required_worker(&self, worker_id: usize) -> Result<&LogicalWorker> {
        self.workers
            .get(worker_id)
            .and_then(Option::as_ref)
            .with_context(|| format!("offline replay selected unknown worker {worker_id}"))
    }

    fn required_worker_mut(&mut self, worker_id: usize) -> Result<&mut LogicalWorker> {
        self.workers
            .get_mut(worker_id)
            .and_then(Option::as_mut)
            .with_context(|| format!("offline replay selected unknown worker {worker_id}"))
    }

    fn scheduler_owner(&self, scheduler_id: usize) -> Result<SchedulerOwner> {
        self.scheduler_owners
            .get(scheduler_id)
            .and_then(|owner| *owner)
            .with_context(|| format!("offline replay selected unknown rank {scheduler_id}"))
    }

    fn refresh_worker(&mut self, worker_id: usize) {
        let ready = self
            .workers
            .get(worker_id)
            .and_then(Option::as_ref)
            .is_some_and(|worker| {
                worker.pending_pass.is_none()
                    && worker.engine.is_ready()
                    && !worker.engine.waiting_for_external_command()
            });
        if ready {
            self.ready_workers.insert(worker_id);
        } else {
            self.ready_workers.remove(&worker_id);
        }
    }

    fn refresh_all_workers(&mut self) {
        for worker_id in 0..self.workers.len() {
            self.refresh_worker(worker_id);
        }
    }

    pub(crate) fn add_worker(&mut self) -> Result<usize> {
        let worker_id = self.workers.len();
        let next_live_worker_count = self
            .live_worker_count
            .checked_add(1)
            .context("live worker count overflow")?;
        let engine = self.factory.build(worker_id)?;
        let dp_size = usize::try_from(self.factory.dp_size())
            .context("native engine dp_size does not fit usize")?;
        let mut scheduler_ids = Vec::with_capacity(dp_size);
        for dp_rank in 0..self.factory.dp_size() {
            let scheduler_id = self.scheduler_owners.len();
            self.scheduler_owners
                .push(Some(SchedulerOwner { worker_id, dp_rank }));
            scheduler_ids.push(scheduler_id);
        }
        self.workers.push(Some(LogicalWorker {
            engine,
            scheduler_ids,
            in_flight_by_rank: vec![BTreeSet::new(); dp_size],
            pending_pass: None,
        }));
        self.live_worker_count = next_live_worker_count;
        self.refresh_worker(worker_id);
        Ok(worker_id)
    }

    fn tombstone_worker(&mut self, worker_id: usize) -> Result<bool> {
        let Some(worker) = self.workers.get(worker_id).and_then(Option::as_ref) else {
            return Ok(false);
        };
        if !worker.engine.is_drained() || worker.pending_pass.is_some() {
            bail!("cannot remove non-drained generalized engine {worker_id}");
        }
        let worker_in_flight = worker.total_in_flight();
        let scheduler_ids = worker.scheduler_ids.clone();
        let next_total_in_flight = self
            .total_in_flight
            .checked_sub(worker_in_flight)
            .context("worker in-flight accounting underflow during removal")?;
        let next_live_worker_count = self
            .live_worker_count
            .checked_sub(1)
            .context("live worker count underflow")?;
        for scheduler_id in &scheduler_ids {
            self.scheduler_owners
                .get(*scheduler_id)
                .context("worker owned an out-of-range scheduler during removal")?;
        }

        // Validate every fallible invariant before mutating the fleet. If a
        // planner-triggered removal fails, Replay can now return the error
        // without leaving a half-tombstoned worker behind.
        self.ready_workers.remove(&worker_id);
        self.workers
            .get_mut(worker_id)
            .context("worker disappeared during removal")?
            .take()
            .context("worker disappeared during removal")?;
        self.total_in_flight = next_total_in_flight;
        self.live_worker_count = next_live_worker_count;
        for scheduler_id in scheduler_ids {
            *self
                .scheduler_owners
                .get_mut(scheduler_id)
                .context("worker owned an out-of-range scheduler during removal")? = None;
        }
        Ok(true)
    }

    pub(crate) fn mark_for_removal(&mut self, worker_id: usize) {
        self.pending_removal.insert(worker_id);
    }

    pub(crate) fn try_remove_drained(&mut self) -> Result<Vec<usize>> {
        let removable = self
            .pending_removal
            .iter()
            .copied()
            .filter(|worker_id| {
                self.workers
                    .get(*worker_id)
                    .and_then(Option::as_ref)
                    .is_none_or(|worker| {
                        worker.pending_pass.is_none()
                            && worker.total_in_flight() == 0
                            && worker.engine.is_drained()
                    })
            })
            .collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(removable.len());
        for worker_id in removable {
            if self
                .tombstone_worker(worker_id)
                .with_context(|| format!("failed to remove drained worker {worker_id}"))?
            {
                removed.push(worker_id);
            }
            self.pending_removal.remove(&worker_id);
        }
        Ok(removed)
    }

    pub(crate) fn apply_target_count(
        &mut self,
        target: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let active_ids = self.active_group_ids();
        let effective = active_ids
            .len()
            .checked_add(self.pending_startup.len())
            .context("non-draining worker count overflow")?;
        let mut added = Vec::new();
        let mut newly_marked = Vec::new();

        if target > effective {
            for _ in 0..(target - effective) {
                let id = self.add_worker().with_context(|| {
                    format!("failed to add worker while scaling from {effective} to {target}")
                })?;
                if self.startup_time_ms.is_some() {
                    self.pending_startup.insert(id);
                    self.ready_workers.remove(&id);
                }
                added.push(id);
            }
        } else if target < effective {
            let excess = effective - target;
            let to_cancel = self
                .pending_startup
                .iter()
                .copied()
                .rev()
                .take(excess)
                .collect::<Vec<_>>();
            for id in &to_cancel {
                self.tombstone_worker(*id)
                    .with_context(|| format!("failed to cancel starting worker {id}"))?;
                self.pending_startup.remove(id);
            }
            for id in active_ids.iter().rev().take(excess - to_cancel.len()) {
                self.mark_for_removal(*id);
                newly_marked.push(*id);
            }
        }

        let removed = self.try_remove_drained()?;
        Ok((added, newly_marked, removed))
    }

    pub(crate) fn active_group_ids(&self) -> Vec<usize> {
        self.workers
            .iter()
            .enumerate()
            .filter(|(worker_id, worker)| {
                worker.is_some()
                    && !self.pending_removal.contains(worker_id)
                    && !self.pending_startup.contains(worker_id)
            })
            .map(|(worker_id, _)| worker_id)
            .collect()
    }

    pub(crate) fn starting_group_ids(&self) -> Vec<usize> {
        self.pending_startup.iter().copied().collect()
    }

    pub(crate) fn draining_group_ids(&self) -> Vec<usize> {
        self.pending_removal.iter().copied().collect()
    }

    pub(crate) fn non_draining_group_count(&self) -> usize {
        self.active_group_ids().len() + self.pending_startup.len()
    }

    pub(crate) fn worker_topology(&self, worker_id: usize) -> Option<WorkerTopology> {
        Some(WorkerTopology {
            worker_id,
            scheduler_ids: self.workers.get(worker_id)?.as_ref()?.scheduler_ids.clone(),
        })
    }

    pub(crate) fn active_topology(&self) -> Vec<WorkerTopology> {
        self.active_group_ids()
            .into_iter()
            .filter_map(|worker_id| self.worker_topology(worker_id))
            .collect()
    }

    pub(crate) fn dp_size(&self) -> u32 {
        self.factory.dp_size()
    }

    pub(crate) fn rank_identity(&self, scheduler_id: usize) -> Option<(usize, u32)> {
        let owner = self
            .scheduler_owners
            .get(scheduler_id)
            .and_then(|owner| *owner)?;
        Some((owner.worker_id, owner.dp_rank))
    }

    pub(crate) fn has_active_workers(&self) -> bool {
        !self.active_group_ids().is_empty()
    }

    pub(crate) fn startup_time_ms(&self) -> Option<f64> {
        self.startup_time_ms
    }

    pub(crate) fn mark_worker_ready(&mut self, worker_id: usize) -> bool {
        let ready = self.pending_startup.remove(&worker_id)
            && self.workers.get(worker_id).is_some_and(Option::is_some);
        if ready {
            self.refresh_worker(worker_id);
        }
        ready
    }

    pub(crate) fn dispatch(
        &mut self,
        scheduler_id: usize,
        request: DirectRequest,
        now_ms: f64,
    ) -> Result<()> {
        let expected = request
            .uuid
            .context("offline replay request must have a UUID before dispatch")?;
        let effects = self.apply_command(
            scheduler_id,
            NativeCommand::Submit(native_request(request)?),
            now_ms,
        )?;
        if !matches!(effects.result, NativeCommandResult::Submitted(id) if id == expected) {
            bail!("native engine returned an unexpected submit result");
        }
        Ok(())
    }

    pub(crate) fn apply_command(
        &mut self,
        scheduler_id: usize,
        command: NativeCommand,
        now_ms: f64,
    ) -> Result<ObservedCommandEffects<Observation::Batch>> {
        let owner = self.scheduler_owner(scheduler_id)?;
        let effects = self
            .required_worker_mut(owner.worker_id)?
            .engine
            .apply_command_effects(SchedulerCommand::new(owner.dp_rank, command), now_ms)?
            .into_by_rank()
            .into_iter()
            .next()
            .context("addressed native command produced no rank effects")?
            .effects;

        let acquired = match effects.result {
            NativeCommandResult::Submitted(request_id)
            | NativeCommandResult::DestinationAccepted { request_id } => Some(request_id),
            NativeCommandResult::Applied | NativeCommandResult::Noop => None,
        };
        self.apply_request_accounting(owner, acquired, &effects.retired_requests)?;
        let engine_events = Observation::observe_native_events(
            self.stage.into(),
            owner.worker_id,
            owner.dp_rank,
            effects.kv_events,
        );
        self.refresh_worker(owner.worker_id);
        Ok(ObservedCommandEffects {
            result: effects.result,
            lifecycle_events: effects.lifecycle_events,
            engine_events,
        })
    }

    fn apply_request_accounting(
        &mut self,
        owner: SchedulerOwner,
        acquired: Option<Uuid>,
        retired: &[Uuid],
    ) -> Result<()> {
        let (before, after) = {
            let worker = self.required_worker_mut(owner.worker_id)?;
            let rank = worker
                .in_flight_by_rank
                .get_mut(owner.dp_rank as usize)
                .context("native command returned an out-of-range DP rank")?;
            let before = rank.len();
            if let Some(request_id) = acquired {
                rank.insert(request_id);
            }
            for request_id in retired {
                rank.remove(request_id);
            }
            (before, rank.len())
        };
        if after >= before {
            self.total_in_flight = self
                .total_in_flight
                .checked_add(after - before)
                .context("offline engine in-flight request count overflow")?;
        } else {
            self.total_in_flight = self
                .total_in_flight
                .checked_sub(before - after)
                .context("offline engine retired more requests than replay tracked")?;
        }
        Ok(())
    }

    pub(crate) fn worker_is_busy(&self, scheduler_id: usize) -> Result<bool> {
        let owner = self.scheduler_owner(scheduler_id)?;
        Ok(self
            .required_worker(owner.worker_id)?
            .pending_pass
            .is_some())
    }

    pub(crate) fn drive_ready(
        &mut self,
        now_ms: f64,
        _collector: Option<&mut TraceCollector>,
    ) -> Result<EngineEffects<Observation::Batch>> {
        while let Some(worker_id) = self.ready_workers.pop_first() {
            let started = {
                let worker = self.required_worker_mut(worker_id)?;
                if worker.pending_pass.is_some()
                    || !worker.engine.is_ready()
                    || worker.engine.waiting_for_external_command()
                {
                    continue;
                }
                worker.engine.execute_pass(now_ms)?
            };
            let Some(started) = started else {
                self.refresh_worker(worker_id);
                continue;
            };

            let pending = PendingPass {
                pass_id: started.pass_id,
                started_at_ms: started.started_at_ms,
                end_ms: started.end_ms,
            };
            self.required_worker_mut(worker_id)?.pending_pass = Some(pending);

            let mut effects: EngineEffects<Observation::Batch> = EngineEffects::default();
            for rank in started.by_rank {
                effects
                    .admissions
                    .extend(rank.effects.admissions.into_iter().map(|admission| {
                        super::AdmissionEvent {
                            uuid: admission.request_id,
                            reused_input_tokens: admission.reused_input_tokens,
                        }
                    }));
                let observed = Observation::observe_native_events(
                    self.stage.into(),
                    worker_id,
                    rank.dp_rank,
                    rank.effects.kv_events,
                );
                effects.pass_start_events.append(observed);
            }

            if started.end_ms > now_ms {
                effects.schedule_completion(
                    started.end_ms,
                    EnginePassCompletion::new(self.stage, worker_id, started.pass_id),
                );
            } else {
                // Completing the pass mutates the authoritative scheduler even
                // when it exposes no effects. Do not surface an empty
                // zero-duration completion as progress: doing so immediately
                // re-arms the still-ready worker and lets the virtual-time
                // driver spin forever at the same timestamp.
                effects.immediate_completions = self
                    .complete_pass(
                        EnginePassCompletion::new(self.stage, worker_id, started.pass_id),
                        now_ms,
                    )?
                    .into_iter()
                    .filter(|payload| payload.progress.made_progress)
                    .collect();
            }
            effects.progress.made_progress = !effects.admissions.is_empty()
                || !effects.pass_start_events.is_empty()
                || !effects.immediate_completions.is_empty()
                || effects.scheduled_completion.is_some();
            return Ok(effects);
        }
        Ok(EngineEffects::default())
    }

    pub(crate) fn on_scheduled_completion(
        &mut self,
        completion: EnginePassCompletion<Observation::Batch>,
        now_ms: f64,
    ) -> Result<Vec<WorkerCompletionPayload<Observation::Batch>>> {
        self.complete_pass(completion, now_ms)
    }

    fn complete_pass(
        &mut self,
        completion: EnginePassCompletion<Observation::Batch>,
        now_ms: f64,
    ) -> Result<Vec<WorkerCompletionPayload<Observation::Batch>>> {
        if completion.stage != self.stage {
            bail!(
                "offline replay completion stage mismatch: expected {:?}, got {:?}",
                self.stage,
                completion.stage
            );
        }
        let pending = self
            .required_worker_mut(completion.worker_id)?
            .pending_pass
            .take()
            .context("offline replay completed a worker with no pass in flight")?;
        if pending.pass_id != completion.pass_id {
            bail!("offline replay generalized-engine pass ID mismatch");
        }
        let completed = self
            .required_worker_mut(completion.worker_id)?
            .engine
            .complete_pass(completion.pass_id, now_ms)?;
        let wall_time_secs = (pending.end_ms - pending.started_at_ms).max(0.0) / 1_000.0;
        let mut payloads = Vec::with_capacity(completed.effects.by_rank.len());
        for rank in completed.effects.into_by_rank() {
            let scheduler_id = self
                .required_worker(completion.worker_id)?
                .scheduler_ids
                .get(rank.dp_rank as usize)
                .copied()
                .context("native completion returned an out-of-range DP rank")?;
            let completed_request_ids = rank
                .effects
                .outputs
                .iter()
                .filter_map(|output| output.completed.then_some(output.request_id))
                .collect::<Vec<_>>();
            self.apply_request_accounting(
                SchedulerOwner {
                    worker_id: completion.worker_id,
                    dp_rank: rank.dp_rank,
                },
                None,
                &completed_request_ids,
            )?;
            payloads.push(lower_completion::<Observation>(
                self.stage,
                completion.worker_id,
                scheduler_id,
                rank.dp_rank,
                wall_time_secs,
                rank.effects,
            ));
        }
        self.refresh_worker(completion.worker_id);
        Ok(payloads)
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.total_in_flight
    }

    pub(crate) fn is_drained(&self) -> bool {
        self.total_in_flight == 0
            && self
                .workers
                .iter()
                .filter_map(Option::as_ref)
                .all(|worker| worker.pending_pass.is_none() && worker.engine.is_drained())
    }

    /// Whether a worker can currently execute a pass that is not waiting on
    /// an external handoff command.
    ///
    /// The replay driver uses this only to distinguish a genuine scheduler
    /// livelock from a legitimate externally blocked handoff when no future
    /// virtual-time event exists.
    pub(crate) fn has_runnable_worker(&self) -> bool {
        !self.ready_workers.is_empty()
    }

    /// Whether Replay still owns requests that the scheduler silently dropped
    /// without terminal effects.
    ///
    /// This is the terminal shape of an effect-free, zero-duration pass (for
    /// example an impossible SGLang admission): no pass or external command
    /// can wake the drained engine, but Replay must still diagnose the lost
    /// terminal signal instead of reporting generic quiescence.
    pub(crate) fn has_orphaned_in_flight(&self) -> bool {
        self.total_in_flight > 0
            && self
                .workers
                .iter()
                .filter_map(Option::as_ref)
                .all(|worker| worker.pending_pass.is_none() && worker.engine.is_drained())
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.live_worker_count
    }
}

fn native_request(request: DirectRequest) -> Result<NativeRequest> {
    Ok(NativeRequest {
        request_id: request
            .uuid
            .context("offline replay request must have a UUID before dispatch")?,
        tokens: request.tokens,
        max_output_tokens: request.max_output_tokens,
        output_token_ids: request.output_token_ids,
    })
}

fn lower_completion<Observation: ReplayEngineObservation>(
    stage: SimulationWorkerStage,
    worker_id: usize,
    scheduler_id: usize,
    dp_rank: u32,
    wall_time_secs: f64,
    effects: NativePassCompletionEffects,
) -> WorkerCompletionPayload<Observation::Batch> {
    let completed_requests = effects
        .outputs
        .iter()
        .filter(|output| output.completed)
        .count();
    let accept_length_output_tokens = effects
        .outputs
        .iter()
        .filter(|output| output.token_id.is_some())
        .count();
    let accept_length_decode_forwards = effects.forward_pass_metrics.num_decode_requests as usize;
    let made_progress = completed_requests > 0
        || !effects.outputs.is_empty()
        || !effects.lifecycle_events.is_empty()
        || !effects.kv_events.is_empty()
        || effects.forward_pass_metrics.num_prefill_requests > 0
        || effects.forward_pass_metrics.num_decode_requests > 0;
    let had_raw_observations = !effects.kv_events.is_empty();
    let engine_events =
        Observation::observe_native_events(stage.into(), worker_id, dp_rank, effects.kv_events);
    WorkerCompletionPayload {
        stage,
        worker_idx: scheduler_id,
        completed_requests,
        output_signals: effects
            .outputs
            .into_iter()
            .map(|output| OutputSignal {
                uuid: output.request_id,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
                handoff_delay_ms: None,
            })
            .collect(),
        lifecycle_events: effects.lifecycle_events,
        engine_events,
        progress: EngineProgress {
            made_progress,
            had_raw_observations,
        },
        fpm: Some(native_fpm(
            dp_rank,
            wall_time_secs,
            effects.forward_pass_metrics,
        )),
        accept_length_output_tokens,
        accept_length_decode_forwards,
    }
}

fn native_fpm(
    dp_rank: u32,
    wall_time_secs: f64,
    fpm: NativeForwardPassMetrics,
) -> ForwardPassSnapshot {
    ForwardPassSnapshot {
        version: 0,
        worker_id: String::new(),
        dp_rank,
        counter_id: 0,
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
        wall_time_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeReplayEngineConfig, NativeReplayFactory, WorkerStage};

    fn component(num_workers: usize, startup_time_ms: Option<f64>) -> EngineComponent {
        let factory = NativeReplayFactory::new()
            .role_factory(
                &NativeReplayEngineConfig::default(),
                WorkerStage::Aggregated,
                false,
            )
            .unwrap();
        EngineComponent::new_with_factory(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            factory,
            num_workers,
            startup_time_ms,
        )
        .unwrap()
    }

    #[test]
    fn canceling_busy_startup_worker_returns_error_without_losing_worker() {
        let mut component = component(1, Some(10.0));
        let (added, _, _) = component.apply_target_count(2).unwrap();
        let starting_worker = added[0];
        let scheduler_id = component
            .worker_topology(starting_worker)
            .unwrap()
            .scheduler_ids[0];
        component
            .dispatch(
                scheduler_id,
                DirectRequest {
                    tokens: vec![1],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(1)),
                    ..Default::default()
                },
                0.0,
            )
            .unwrap();

        let error = component.apply_target_count(1).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to cancel starting worker")
        );
        assert_eq!(component.starting_group_ids(), vec![starting_worker]);
        assert!(component.worker_topology(starting_worker).is_some());
        assert_eq!(component.worker_count(), 2);
    }

    #[test]
    fn drained_removal_error_preserves_pending_worker() {
        let mut component = component(1, None);
        component.mark_for_removal(0);
        component.live_worker_count = 0;

        let error = component.try_remove_drained().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to remove drained worker 0")
        );
        assert_eq!(component.draining_group_ids(), vec![0]);
        assert!(component.worker_topology(0).is_some());
    }
}

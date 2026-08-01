// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::components::ReplayWorkerCore;
use super::progress::ReplayProgress;
use crate::artifact::{ReplayArtifactRequest, ReplayArtifactSink};
use crate::loadgen::WorkloadDriver;
use crate::native::NativeRoleFactory;
use crate::protocol::DirectRequest;
use crate::replay::{ReplayTerminalStatus, TraceCollector};
use anyhow::{Context, bail};
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub(super) enum SingleReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

// SGLang may intentionally retry 600 same-timestamp passes while reducing its
// output reservation ratio before an otherwise valid request can be admitted.
const MAX_CONSECUTIVE_NO_PROGRESS_PASSES: usize = 1024;

pub(super) struct SingleRuntime {
    current_time_ms: f64,
    admission: AdmissionSource,
    worker: ReplayWorkerCore,
    collector: TraceCollector,
    mode: SingleReplayMode,
    progress: ReplayProgress,
    consecutive_no_progress_passes: usize,
    artifact_sink: Option<ReplayArtifactSink>,
    /// Optional cap on simulated wall-clock time. When set, `run()` exits
    /// gracefully once `current_time_ms` exceeds this cap, leaving any
    /// in-flight requests as incomplete in the report.
    max_sim_time_ms: Option<f64>,
}

impl SingleRuntime {
    pub(super) fn new(
        factory: NativeRoleFactory,
        pending: VecDeque<DirectRequest>,
        mode: SingleReplayMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(factory, AdmissionSource::Requests(pending), mode)
    }

    pub(super) fn new_workload(
        factory: NativeRoleFactory,
        driver: WorkloadDriver,
        mode: SingleReplayMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(factory, AdmissionSource::Workload(driver), mode)
    }

    fn new_with_source(
        factory: NativeRoleFactory,
        admission: AdmissionSource,
        mode: SingleReplayMode,
    ) -> anyhow::Result<Self> {
        let total_requests = match &admission {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        };
        // The single-worker runtime has exactly one (decode) worker for the
        // whole run and no event loop to integrate, so declare a static count;
        // `finish()` derives worker-seconds as 1 × duration_s.
        let mut collector = TraceCollector::default();
        collector.set_static_worker_count(0, 1);
        collector.set_gpus_per_worker(0, factory.gpus_per_worker()?);
        Ok(Self {
            current_time_ms: 0.0,
            admission,
            worker: ReplayWorkerCore::new(factory)?,
            collector,
            mode,
            progress: ReplayProgress::new(total_requests, "offline replay"),
            consecutive_no_progress_passes: 0,
            artifact_sink: None,
            max_sim_time_ms: None,
        })
    }

    /// Toggle per-request record capture on the underlying collector. When
    /// `true`, the final `TraceSimulationReport` returned from `run()` will
    /// have `per_request` populated. Default `false` (cheap).
    pub(crate) fn with_per_request_records(mut self, capture: bool) -> Self {
        self.collector.set_capture_per_request(capture);
        self
    }

    /// Cap the simulated wall-clock duration. After construction, call this to
    /// have `run()` stop gracefully once the simulated clock would exceed
    /// `ms`. Pass `None` to run to natural completion (the default).
    ///
    /// max_sim_time_ms is a **soft cap** on the scheduling loop, not a hard truncation
    /// of recorded work. When the next scheduled simulated timestamp would
    /// exceed the cap, the loop exits, but worker passes already in flight
    /// complete normally — even if their token timestamps land past
    /// `ms`. Requests that hadn't received their first token before the cap
    /// fired stay in the report as incomplete (`first_token_ms = None`,
    /// `e2e_latency_ms = None`). `report.duration_ms` may exceed `ms` by up
    /// to one in-flight pass's duration. Enforcing a precise cap here would
    /// require plumbing a deadline into the worker / engine core; not worth
    /// it for the calibration use case this exists to serve.
    pub(super) fn with_max_sim_time_ms(mut self, ms: Option<f64>) -> Self {
        self.max_sim_time_ms = ms;
        self
    }

    pub(super) fn with_artifact_sink(mut self, sink: ReplayArtifactSink) -> Self {
        self.artifact_sink = Some(sink);
        self
    }

    fn enqueue_trace_arrivals(&mut self) -> anyhow::Result<()> {
        let mut ready_requests = Vec::new();
        // A pass that started before the soft cap may finish after it. Admit
        // arrivals authored at or before the cap, but never pull later work
        // into the simulated system merely because that pass advanced the
        // single-worker clock beyond the cutoff.
        let ready_at_ms = self.max_sim_time_ms.map_or(self.current_time_ms, |cap_ms| {
            self.current_time_ms.min(cap_ms)
        });
        let artifact_sink = self.artifact_sink.clone();
        match &mut self.admission {
            AdmissionSource::Requests(pending) => {
                while let Some(next_arrival_ms) = pending
                    .front()
                    .and_then(|request| request.arrival_timestamp_ms)
                {
                    if next_arrival_ms > ready_at_ms {
                        break;
                    }

                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is available");
                    let arrival_ms = request
                        .arrival_timestamp_ms
                        .expect("trace replay requests must have an arrival timestamp");
                    if let Some(sink) = &artifact_sink {
                        sink.record_request(ReplayArtifactRequest {
                            request_id: request.uuid.context(
                                "artifact replay request must have a UUID before admission",
                            )?,
                            observed_at_ms: self.current_time_ms,
                            scheduled_ready_at_ms: arrival_ms,
                            input_length: request.tokens.len(),
                            output_length: request.max_output_tokens,
                            replay_hashes: None,
                        })?;
                    }
                    ready_requests.push((request, arrival_ms));
                }
            }
            AdmissionSource::Workload(driver) => {
                for ready in driver.pop_ready(ready_at_ms, usize::MAX) {
                    if let Some(sink) = &artifact_sink {
                        sink.record_request(ReplayArtifactRequest {
                            request_id: ready.request_uuid,
                            observed_at_ms: self.current_time_ms,
                            scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                            input_length: ready.request.tokens.len(),
                            output_length: ready.request.max_output_tokens,
                            replay_hashes: ready.replay_hashes,
                        })?;
                    }
                    ready_requests.push((ready.request, ready.scheduled_ready_at_ms));
                }
            }
        }

        for (request, arrival_ms) in ready_requests {
            self.record_arrival(request, arrival_ms)?;
        }
        Ok(())
    }

    fn enqueue_concurrency_arrivals(&mut self, max_in_flight: usize) -> anyhow::Result<()> {
        let available = max_in_flight.saturating_sub(self.worker.num_requests());
        let mut ready_requests = Vec::new();
        let artifact_sink = self.artifact_sink.clone();

        match &mut self.admission {
            AdmissionSource::Requests(pending) => {
                for _ in 0..available {
                    let Some(mut request) = pending.pop_front() else {
                        break;
                    };
                    request.arrival_timestamp_ms = Some(self.current_time_ms);
                    if let Some(sink) = &artifact_sink {
                        sink.record_request(ReplayArtifactRequest {
                            request_id: request.uuid.context(
                                "artifact replay request must have a UUID before admission",
                            )?,
                            observed_at_ms: self.current_time_ms,
                            scheduled_ready_at_ms: self.current_time_ms,
                            input_length: request.tokens.len(),
                            output_length: request.max_output_tokens,
                            replay_hashes: None,
                        })?;
                    }
                    ready_requests.push(request);
                }
            }
            AdmissionSource::Workload(driver) => {
                for ready in driver.pop_ready(self.current_time_ms, available) {
                    if let Some(sink) = &artifact_sink {
                        sink.record_request(ReplayArtifactRequest {
                            request_id: ready.request_uuid,
                            observed_at_ms: self.current_time_ms,
                            scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                            input_length: ready.request.tokens.len(),
                            output_length: ready.request.max_output_tokens,
                            replay_hashes: ready.replay_hashes,
                        })?;
                    }
                    ready_requests.push(ready.request);
                }
            }
        }

        for request in ready_requests {
            self.record_arrival(request, self.current_time_ms)?;
        }
        Ok(())
    }

    fn record_arrival(&mut self, request: DirectRequest, arrival_ms: f64) -> anyhow::Result<Uuid> {
        let input_length = request.tokens.len();
        let output_length = request.max_output_tokens;
        let replay_context = request.replay_context.clone();
        let uuid = self.worker.receive(request, self.current_time_ms)?;
        self.collector
            .on_arrival(uuid, arrival_ms, input_length, output_length);
        if let Some(context) = replay_context.as_ref() {
            self.collector.on_request_context(uuid, context);
        }
        // Aggregated replay has one logical pool. Match the general runtime's
        // assignment record so callers cannot observe which execution path
        // was selected for a one-worker deployment.
        self.collector.on_decode_assigned(uuid, 0);
        Ok(uuid)
    }

    fn is_done(&self) -> bool {
        self.worker.is_empty()
            && match &self.admission {
                AdmissionSource::Requests(pending) => pending.is_empty(),
                AdmissionSource::Workload(driver) => driver.is_drained(),
            }
    }

    fn advance_to_next_trace_arrival(&mut self) -> anyhow::Result<bool> {
        let next_arrival_ms = match &mut self.admission {
            AdmissionSource::Requests(pending) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            AdmissionSource::Workload(driver) => driver.next_ready_time_ms(),
        };
        let Some(next_arrival_ms) = next_arrival_ms else {
            bail!("trace replay reached an idle state without a pending arrival");
        };
        if self
            .max_sim_time_ms
            .is_some_and(|cap_ms| next_arrival_ms > cap_ms)
        {
            return Ok(false);
        }
        self.current_time_ms = next_arrival_ms;
        Ok(true)
    }

    fn drive_worker(&mut self, admit_arrivals_between_steps: bool) -> anyhow::Result<()> {
        let pass_start_ms = self.current_time_ms;
        let requests_before = self.worker.num_requests();
        let pass = self
            .worker
            .execute_pass(&mut self.collector, self.current_time_ms)?;
        self.current_time_ms = pass.end_ms;
        if let Some(sink) = &self.artifact_sink {
            sink.record_pass_kv_events(
                pass_start_ms,
                self.current_time_ms,
                &pass.pass_start_kv_events,
                &pass.pass_end_kv_events,
            )?;
            sink.record_outputs(self.current_time_ms, &pass.output_signals)?;
        }
        let made_progress = self.current_time_ms > pass_start_ms
            || self.worker.num_requests() < requests_before
            || pass.completed_requests > 0
            || !pass.admissions.is_empty()
            || !pass.output_signals.is_empty()
            || pass.has_kv_events();
        if let AdmissionSource::Workload(driver) = &mut self.admission {
            for signal in &pass.output_signals {
                if let Some(token_id) = signal.token_id {
                    driver
                        .on_output_token(signal.uuid, token_id)
                        .with_context(|| {
                            format!(
                                "failed to record output token for workload request {}",
                                signal.uuid
                            )
                        })?;
                }
                if signal.completed {
                    driver
                        .on_terminal(signal.uuid, self.current_time_ms, signal.rejected)
                        .with_context(|| {
                            format!(
                                "failed to process terminal signal for workload request {}",
                                signal.uuid
                            )
                        })?;
                }
            }
        }
        let completed_requests = pass
            .output_signals
            .iter()
            .filter(|signal| signal.completed)
            .count();
        for signal in pass.output_signals.iter().filter(|signal| signal.completed) {
            let status = if signal.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            self.collector
                .on_terminal(signal.uuid, self.current_time_ms, status);
        }
        for _ in 0..completed_requests {
            self.progress.inc_completed();
        }
        if admit_arrivals_between_steps {
            self.enqueue_trace_arrivals()?;
        }
        if made_progress {
            self.consecutive_no_progress_passes = 0;
            return Ok(());
        }

        self.consecutive_no_progress_passes += 1;
        if self.consecutive_no_progress_passes >= MAX_CONSECUTIVE_NO_PROGRESS_PASSES
            && !self.worker.is_empty()
        {
            bail!(
                "offline replay detected an effect-free zero-duration pass with {} in-flight requests remaining",
                self.worker.num_requests()
            );
        }
        Ok(())
    }

    pub(super) fn run(mut self) -> anyhow::Result<TraceCollector> {
        if let Some(cap_ms) = self.max_sim_time_ms
            && (!cap_ms.is_finite() || cap_ms < 0.0)
        {
            anyhow::bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
        }
        while !self.is_done() {
            if let Some(cap_ms) = self.max_sim_time_ms
                && self.current_time_ms > cap_ms
            {
                break;
            }
            match self.mode {
                SingleReplayMode::Trace => {
                    self.enqueue_trace_arrivals()?;
                    if self.worker.is_empty() {
                        if !self.advance_to_next_trace_arrival()? {
                            break;
                        }
                        self.enqueue_trace_arrivals()?;
                        continue;
                    }
                    self.drive_worker(true)?;
                }
                SingleReplayMode::Concurrency { max_in_flight } => {
                    self.enqueue_concurrency_arrivals(max_in_flight)?;
                    if self.worker.is_empty() {
                        if self.is_done() {
                            break;
                        }
                        if !self.advance_to_next_trace_arrival()? {
                            break;
                        }
                        continue;
                    }
                    self.drive_worker(false)?;
                }
            }
        }

        self.progress.finish();
        Ok(self.collector)
    }
}

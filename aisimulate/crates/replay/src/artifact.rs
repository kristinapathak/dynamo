// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-neutral observations captured from a shared Replay execution.

use std::sync::{Arc, Mutex};

use aisimulate_engine::NativeKvEvent;
use uuid::Uuid;

use crate::loadgen::ReplayRequestHashes;
use crate::{ReplayError, ReplayResult};

/// Timestamp policy used when rendering native KV observations as replay
/// artifacts.
///
/// `Native` preserves each backend's publication boundary. The two explicit
/// variants exist for parity fixtures that intentionally normalize all events
/// to one side of a pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplayArtifactKvEventVisibility {
    #[default]
    Native,
    PassStart,
    PassEnd,
}

/// One request released by Replay's workload source.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactRequest {
    pub request_id: Uuid,
    /// Virtual time at which Replay made the request visible to the engine.
    pub observed_at_ms: f64,
    /// Workload-authored ready time, which can precede `observed_at_ms` while
    /// the synchronous single-worker runtime is completing a pass.
    pub scheduled_ready_at_ms: f64,
    pub input_length: usize,
    pub output_length: usize,
    pub replay_hashes: Option<ReplayRequestHashes>,
}

/// One client-visible output released at a pass-completion boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactOutput {
    pub request_id: Uuid,
    pub token_id: Option<u32>,
    pub completed: bool,
    pub rejected: bool,
    pub observed_at_ms: f64,
}

/// One native G1 observation at the timestamp selected for the artifact run.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactKvEvent {
    pub event: NativeKvEvent,
    pub observed_at_ms: f64,
}

/// Optional detailed observations produced by the same virtual-clock/pass
/// loop that generates the normal replay report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayArtifacts {
    pub requests: Vec<ReplayArtifactRequest>,
    pub outputs: Vec<ReplayArtifactOutput>,
    pub kv_events: Vec<ReplayArtifactKvEvent>,
}

/// Shared sink retained by the caller while [`crate::Replayer`] owns the run.
///
/// This is intentionally a concrete Replay-owned sink instead of a plugin ABI:
/// it carries only neutral request/output/native-KV values and introduces no
/// dependency on an adapter runtime.
#[derive(Debug, Clone)]
pub(crate) struct ReplayArtifactSink {
    visibility: ReplayArtifactKvEventVisibility,
    artifacts: Arc<Mutex<ReplayArtifacts>>,
}

impl ReplayArtifactSink {
    pub(crate) fn new(visibility: ReplayArtifactKvEventVisibility) -> Self {
        Self {
            visibility,
            artifacts: Arc::new(Mutex::new(ReplayArtifacts::default())),
        }
    }

    pub(crate) fn record_request(&self, request: ReplayArtifactRequest) -> ReplayResult<()> {
        self.lock()?.requests.push(request);
        Ok(())
    }

    pub(crate) fn record_outputs(
        &self,
        observed_at_ms: f64,
        outputs: &[crate::protocol::OutputSignal],
    ) -> ReplayResult<()> {
        self.lock()?
            .outputs
            .extend(outputs.iter().map(|output| ReplayArtifactOutput {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
                observed_at_ms,
            }));
        Ok(())
    }

    pub(crate) fn record_pass_kv_events(
        &self,
        pass_start_ms: f64,
        pass_end_ms: f64,
        pass_start_events: &[NativeKvEvent],
        pass_end_events: &[NativeKvEvent],
    ) -> ReplayResult<()> {
        let (start_timestamp_ms, end_timestamp_ms) = match self.visibility {
            ReplayArtifactKvEventVisibility::Native => (pass_start_ms, pass_end_ms),
            ReplayArtifactKvEventVisibility::PassStart => (pass_start_ms, pass_start_ms),
            ReplayArtifactKvEventVisibility::PassEnd => (pass_end_ms, pass_end_ms),
        };
        let mut artifacts = self.lock()?;
        artifacts
            .kv_events
            .extend(
                pass_start_events
                    .iter()
                    .cloned()
                    .map(|event| ReplayArtifactKvEvent {
                        event,
                        observed_at_ms: start_timestamp_ms,
                    }),
            );
        artifacts
            .kv_events
            .extend(
                pass_end_events
                    .iter()
                    .cloned()
                    .map(|event| ReplayArtifactKvEvent {
                        event,
                        observed_at_ms: end_timestamp_ms,
                    }),
            );
        Ok(())
    }

    pub(crate) fn take(&self) -> ReplayResult<ReplayArtifacts> {
        Ok(std::mem::take(&mut *self.lock()?))
    }

    fn lock(&self) -> ReplayResult<std::sync::MutexGuard<'_, ReplayArtifacts>> {
        self.artifacts.lock().map_err(|_| {
            ReplayError::Invariant("replay artifact sink lock was poisoned".to_string())
        })
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_engine::{NativeEngineConfig, NativeTimingModelConfig};
use aisimulate_replay::loadgen::{SessionTrace, Trace, TurnTrace};
use aisimulate_replay::{
    CURRENT_REPLAY_SPEC_VERSION, NativeReplayEngineConfig, NativeReplayFactory, ProviderSpec,
    ReplayAdapters, ReplayArtifactKvEventVisibility, ReplayRuntimeInput, ReplaySpec,
    ReplayTopology, Replayer, WorkerPoolSpec,
};

fn artifact_spec(workers: usize) -> ReplaySpec {
    let engine = NativeReplayEngineConfig {
        rank: NativeEngineConfig {
            num_gpu_blocks: 64,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: NativeTimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 2.0,
            },
            ..NativeEngineConfig::default()
        },
        ..NativeReplayEngineConfig::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms: 0.0,
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: false,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn workload() -> aisimulate_replay::loadgen::WorkloadDriver {
    Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "artifact-session".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 8,
                max_output_tokens: 2,
                hash_ids: vec![11, 12],
                ..TurnTrace::default()
            }],
        }],
    }
    .into_trace_driver_with_block_size(4)
    .unwrap()
}

#[test]
fn artifact_capture_uses_the_shared_single_worker_replayer_loop() {
    let (report, artifacts) = Replayer::new(artifact_spec(1), NativeReplayFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(workload()))
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();

    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(artifacts.requests.len(), 1);
    assert_eq!(artifacts.requests[0].observed_at_ms, 0.0);
    assert_eq!(artifacts.requests[0].scheduled_ready_at_ms, 0.0);
    assert_eq!(artifacts.requests[0].input_length, 8);
    assert_eq!(artifacts.requests[0].output_length, 2);
    assert!(artifacts.requests[0].replay_hashes.is_some());
    assert_eq!(artifacts.outputs.len(), 2);
    assert!(artifacts.outputs.last().unwrap().completed);
    assert!(
        artifacts
            .outputs
            .windows(2)
            .all(|pair| pair[0].observed_at_ms <= pair[1].observed_at_ms)
    );
    assert!(!artifacts.kv_events.is_empty());
    assert!(
        artifacts
            .kv_events
            .windows(2)
            .all(|pair| pair[0].observed_at_ms <= pair[1].observed_at_ms)
    );
}

#[test]
fn artifact_capture_rejects_a_topology_that_would_use_another_runtime() {
    let error = Replayer::new(artifact_spec(2), NativeReplayFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(workload()))
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap_err();

    assert!(
        error.to_string().contains("one logical DP1 worker"),
        "{error}"
    );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;

use aisimulate_engine::{NativeBackend, NativeEngineConfig, NativeTimingModelConfig};
use aisimulate_replay::{
    AggregatedRoundRobinPlacement, NativeReplayEngineConfig, NativeReplayFactory, NoEngineEvents,
    NoReplayMetadata, PoolRoundRobinPlacement, ProviderSpec, ReplayAdapters, ReplayComposition,
    ReplayRequest, ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec,
    ReplayTopology, Replayer, TraceSimulationReport, WorkerPoolSpec, WorkerTopology,
    run_native_replay,
};

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens,
        input_token_ids: None,
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

fn aggregated_spec(
    backend: NativeBackend,
    workers: usize,
    startup_delay_ms: f64,
    requests: Vec<ReplayRequest>,
) -> ReplaySpec {
    let rank = NativeEngineConfig {
        num_gpu_blocks: 64,
        block_size: 4,
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        timing_model: NativeTimingModelConfig::Fixed {
            prefill_ms: 10.0,
            decode_ms: 2.0,
        },
        ..NativeEngineConfig::for_backend(backend)
    };
    ReplaySpec {
        version: 1,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms,
            },
        },
        engine: serde_json::to_value(NativeReplayEngineConfig {
            rank,
            ..NativeReplayEngineConfig::default()
        })
        .unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests,
    }
}

fn assert_reports_equal(left: &TraceSimulationReport, right: &TraceSimulationReport) {
    // Replay execution wall time is intentionally diagnostic rather than a
    // virtual-time semantic. Zeroing it also zeroes the two wall-derived
    // processed-token rates in the serialized report.
    let left_summary = left.clone().with_wall_time_ms(0.0);
    let right_summary = right.clone().with_wall_time_ms(0.0);
    assert_eq!(
        serde_json::to_value(left_summary).unwrap(),
        serde_json::to_value(right_summary).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&left.per_request).unwrap(),
        serde_json::to_value(&right.per_request).unwrap()
    );
}

struct GeneralRoundRobinComposition {
    scaling: Option<Box<dyn ReplayScalingPolicy>>,
}

impl GeneralRoundRobinComposition {
    fn fixed_capacity() -> Self {
        Self { scaling: None }
    }

    fn with_scaling(scaling: impl ReplayScalingPolicy + 'static) -> Self {
        Self {
            scaling: Some(Box::new(scaling)),
        }
    }
}

impl ReplayComposition for GeneralRoundRobinComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(
        &mut self,
    ) -> aisimulate_replay::ReplayResult<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling.take())
    }
}

fn run_general_aggregated(spec: ReplaySpec) -> TraceSimulationReport {
    Replayer::with_composition(
        spec,
        NativeReplayFactory::new(),
        GeneralRoundRobinComposition::fixed_capacity(),
    )
    .unwrap()
    .run()
    .unwrap()
}

#[test]
fn single_worker_fast_path_matches_general_aggregated_runtime() {
    let requests = vec![
        request("first", 0.0, 8, 3),
        request("second", 0.0, 5, 2),
        request("third", 17.0, 7, 4),
    ];

    for backend in [
        NativeBackend::Vllm,
        NativeBackend::Sglang,
        NativeBackend::Trtllm,
    ] {
        let spec = aggregated_spec(backend, 1, 0.0, requests.clone());
        let fast_path = run_native_replay(spec.clone()).unwrap();
        let general = run_general_aggregated(spec);

        assert_reports_equal(&fast_path, &general);
        assert_eq!(fast_path.request_counts.completed_requests, 3);
    }
}

#[test]
fn max_sim_time_is_the_same_soft_cap_for_fast_and_general_aggregated_runtimes() {
    let requests = (0..5)
        .map(|index| {
            request(
                &format!("request-{index}"),
                f64::from(index) * 1_000.0,
                4,
                2,
            )
        })
        .collect();
    let mut capped = aggregated_spec(NativeBackend::Vllm, 1, 0.0, requests);
    capped.max_sim_time_ms = Some(2_500.0);

    let fast_path = run_native_replay(capped.clone()).unwrap();
    let general = run_general_aggregated(capped);

    assert_reports_equal(&fast_path, &general);
    assert_eq!(fast_path.request_counts.num_requests, 3);
    assert_eq!(fast_path.request_counts.completed_requests, 3);
    assert!(fast_path.throughput.duration_ms <= 2_500.0);
}

#[derive(Debug, Clone, PartialEq)]
struct ScalingObservation {
    now_ms: f64,
    active: Vec<usize>,
    starting: Vec<usize>,
    draining: Vec<usize>,
}

struct ScaleUpThenDown {
    step: usize,
    observations: Rc<RefCell<Vec<ScalingObservation>>>,
}

impl ReplayScalingPolicy for ScaleUpThenDown {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(100.0)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        self.observations.borrow_mut().push(ScalingObservation {
            now_ms: snapshot.now_ms,
            active: snapshot.active_decode_ids,
            starting: snapshot.starting_decode_ids,
            draining: snapshot.draining_decode_ids,
        });
        let decision = match self.step {
            0 => ReplayScalingDecision {
                target_decode: Some(2),
                next_tick_ms: Some(200.0),
                ..ReplayScalingDecision::default()
            },
            1 => ReplayScalingDecision {
                next_tick_ms: Some(300.0),
                ..ReplayScalingDecision::default()
            },
            2 => ReplayScalingDecision {
                target_decode: Some(1),
                next_tick_ms: Some(400.0),
                ..ReplayScalingDecision::default()
            },
            3 => ReplayScalingDecision::default(),
            _ => panic!("unexpected scaling tick {}", self.step),
        };
        self.step += 1;
        Ok(decision)
    }
}

#[test]
fn scaling_honors_startup_delay_then_scales_the_ready_worker_back_down() {
    let observations = Rc::new(RefCell::new(Vec::new()));
    let policy = ScaleUpThenDown {
        step: 0,
        observations: Rc::clone(&observations),
    };
    let mut spec = aggregated_spec(
        NativeBackend::Vllm,
        1,
        200.0,
        vec![request("long-running", 0.0, 4, 2)],
    );
    let mut engine: NativeReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    engine.rank.timing_model = NativeTimingModelConfig::Fixed {
        prefill_ms: 1_000.0,
        decode_ms: 1_000.0,
    };
    spec.engine = serde_json::to_value(engine).unwrap();
    spec.adapters.scaling = ProviderSpec {
        provider: "scripted_test_policy".to_string(),
        config: serde_json::Value::Null,
    };

    let report = Replayer::with_composition(
        spec,
        NativeReplayFactory::new(),
        GeneralRoundRobinComposition::with_scaling(policy),
    )
    .unwrap()
    .run()
    .unwrap();

    assert_eq!(
        *observations.borrow(),
        vec![
            ScalingObservation {
                now_ms: 100.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 200.0,
                active: vec![0],
                starting: vec![1],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 300.0,
                active: vec![0, 1],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 400.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
        ]
    );
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.throughput.duration_ms, 3_000.0);
    assert!((report.throughput.decode_worker_seconds - 3.2).abs() < 1e-9);
}

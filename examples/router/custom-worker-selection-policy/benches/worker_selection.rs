// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
use dynamo_kv_router::{
    DefaultWorkerSelector, KvRouterConfig, SchedulingRequest, WorkerConfigLike,
    WorkerLoadProjection, WorkerSelector,
};
use dynamo_sglang_cache_load_policy::{SglangCacheLoadConfig, sglang_cache_load_policy};

struct BenchWorkerConfig;

impl WorkerConfigLike for BenchWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(131_072)
    }
}

fn fixture(worker_count: usize) -> (HashMap<WorkerId, BenchWorkerConfig>, SchedulingRequest) {
    let mut workers = HashMap::with_capacity(worker_count);
    let mut effective_overlap_blocks = HashMap::with_capacity(worker_count);
    let mut worker_loads = HashMap::with_capacity(worker_count);

    for worker_id in 0..worker_count as WorkerId {
        let worker = WorkerWithDpRank::from_worker_id(worker_id);
        workers.insert(worker_id, BenchWorkerConfig);
        effective_overlap_blocks.insert(worker, (worker_id % 32) as f64);
        worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_requests: worker_id as usize % 17,
                ..Default::default()
            },
        );
    }

    let request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 2_048,
        lora_name: None,
        expected_output_tokens: Some(256),
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: false,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_id: None,
        overlap: OverlapSignals {
            effective_overlap_blocks,
            ..Default::default()
        },
        shared_cache_hits: None,
        worker_loads: worker_loads.into_iter().collect(),
        resp_tx: None,
    };

    (workers, request)
}

fn worker_selection(c: &mut Criterion) {
    for temperature in [0.0, 0.7] {
        let mut group = c.benchmark_group(format!("worker_selection/temperature_{temperature}"));
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(50);

        for worker_count in [2, 32, 1_024, 10_000] {
            let (workers, request) = fixture(worker_count);
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let default = DefaultWorkerSelector::new(Some(config.clone()), "prefill");
            let custom =
                sglang_cache_load_policy(config, "prefill", SglangCacheLoadConfig::default());

            group.throughput(Throughput::Elements(worker_count as u64));
            group.bench_with_input(
                BenchmarkId::new("default", worker_count),
                &worker_count,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            default
                                .select_worker(
                                    black_box(&workers),
                                    black_box(&request),
                                    request.eligibility(),
                                    black_box(16),
                                )
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("sglang_cache_load", worker_count),
                &worker_count,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            custom
                                .select_worker(
                                    black_box(&workers),
                                    black_box(&request),
                                    request.eligibility(),
                                    black_box(16),
                                )
                                .unwrap(),
                        )
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, worker_selection);
criterion_main!(benches);

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang Model Gateway-style cache/load worker selection for Dynamo, using Dynamo's exact KV
//! overlap and active-request signals.

use dynamo_kv_router::{
    KvRouterConfig, WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// SGLang Model Gateway's cache/load routing thresholds.
#[derive(Clone, Copy, Debug)]
pub struct SglangCacheLoadConfig {
    pub cache_threshold: f64,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f64,
}

impl Default for SglangCacheLoadConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
        }
    }
}

/// Scores each eligible worker by its active request count.
pub struct ActiveRequestScorer;

impl WorkerScorer for ActiveRequestScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        Ok(candidate
            .load()
            .expect("active-request scorer declared load inputs")
            .active_requests() as f64)
    }
}

/// Uses shortest-queue routing when load is imbalanced and cache affinity otherwise.
pub struct SglangCacheLoadPicker {
    config: SglangCacheLoadConfig,
}

impl SglangCacheLoadPicker {
    pub fn new(config: SglangCacheLoadConfig) -> Self {
        Self { config }
    }
}

impl WorkerPicker for SglangCacheLoadPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let Some((first, rest)) = candidates.split_first() else {
            return Err(WorkerSelectionPolicyError::rejected(
                "SGLang cache/load picker requires at least one candidate",
            ));
        };
        let cache = input.cache().expect("picker declared cache inputs");
        let mut min_load = (0, first.cost());
        let mut max_load = first.cost();
        let mut max_overlap = (0, cache[0].effective_overlap_blocks());

        for (row, (candidate, cache)) in rest.iter().zip(&cache[1..]).enumerate() {
            let row = row + 1;
            if candidate.cost() < min_load.1 {
                min_load = (row, candidate.cost());
            }
            max_load = max_load.max(candidate.cost());
            if cache.effective_overlap_blocks() > max_overlap.1 {
                max_overlap = (row, cache.effective_overlap_blocks());
            }
        }

        let imbalanced = max_load - min_load.1 > self.config.balance_abs_threshold as f64
            && max_load > min_load.1 * self.config.balance_rel_threshold;
        if imbalanced {
            return Ok(min_load.0);
        }

        let match_rate = max_overlap.1 / context.request_blocks() as f64;
        Ok(if match_rate > self.config.cache_threshold {
            max_overlap.0
        } else {
            min_load.0
        })
    }
}

pub fn sglang_cache_load_policy(
    kv_router_config: KvRouterConfig,
    worker_type: &'static str,
    config: SglangCacheLoadConfig,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        kv_router_config,
        worker_type,
        vec![Box::new(ActiveRequestScorer)],
        Box::new(SglangCacheLoadPicker::new(config)),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerConfigLike, WorkerLoadProjection, WorkerSelector,
    };

    use super::*;

    struct TestWorkerConfig;

    impl WorkerConfigLike for TestWorkerConfig {
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
            None
        }
    }

    fn select(config: SglangCacheLoadConfig, cached_blocks: f64) -> u64 {
        let cached = WorkerWithDpRank::from_worker_id(0);
        let least_loaded = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([(0, TestWorkerConfig), (1, TestWorkerConfig)]);
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 160,
            lora_name: None,
            expected_output_tokens: None,
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
                effective_overlap_blocks: HashMap::from([(cached, cached_blocks)]),
                ..Default::default()
            },
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        request.worker_loads.insert(
            cached,
            WorkerLoadProjection {
                active_requests: 4,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            least_loaded,
            WorkerLoadProjection {
                active_requests: 1,
                ..Default::default()
            },
        );
        let policy = sglang_cache_load_policy(KvRouterConfig::default(), "prefill", config);

        policy
            .select_worker(&workers, &request, request.eligibility(), 16)
            .unwrap()
            .worker
            .worker_id
    }

    #[test]
    fn matches_sglang_cache_load_decision_boundaries() {
        let balanced = SglangCacheLoadConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 3,
            balance_rel_threshold: 4.0,
        };
        assert_eq!(select(balanced, 8.0), 0, "cache wins while balanced");
        assert_eq!(select(balanced, 5.0), 1, "cache threshold is strict");

        let imbalanced = SglangCacheLoadConfig {
            balance_abs_threshold: 2,
            balance_rel_threshold: 2.0,
            ..Default::default()
        };
        assert_eq!(select(imbalanced, 8.0), 1, "load wins when imbalanced");
    }
}

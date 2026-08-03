// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf};

use dynamo_kv_router::KvRouterConfig;
use dynamo_llm::{
    entrypoint::{EngineConfig, HttpFrontend, RouterConfig},
    local_model::LocalModelBuilder,
};
use dynamo_runtime::{DistributedRuntime, Runtime, Worker, logging, pipeline::RouterMode};
use dynamo_sglang_cache_load_policy::{SglangCacheLoadConfig, sglang_cache_load_policy};

fn main() -> anyhow::Result<()> {
    logging::init();
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: frontend MODEL_PATH MODEL_NAME"))?;
    let model_name = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: frontend MODEL_PATH MODEL_NAME"))?;
    if args.next().is_some() {
        anyhow::bail!("usage: frontend MODEL_PATH MODEL_NAME");
    }

    Worker::from_settings()?.execute(move |runtime| run(runtime, model_path, model_name))
}

async fn run(runtime: Runtime, model_path: String, model_name: String) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime).await?;
    let kv_router_config = KvRouterConfig::default();
    let mut model = LocalModelBuilder::default();
    model
        .model_path(PathBuf::from(model_path))
        .model_name(Some(model_name))
        .namespace(Some(
            env::var("DYN_NAMESPACE").unwrap_or_else(|_| "dynamo".to_string()),
        ))
        .http_port(
            env::var("DYN_HTTP_PORT")
                .ok()
                .map(|port| port.parse())
                .transpose()?
                .unwrap_or(8000),
        )
        .kv_cache_block_size(
            env::var("DYN_KV_CACHE_BLOCK_SIZE")
                .ok()
                .map(|block_size| block_size.parse())
                .transpose()?,
        )
        .router_config(Some(RouterConfig::new(RouterMode::KV, kv_router_config)));

    HttpFrontend::default()
        .worker_selection_policy_factory(|config, worker_type| {
            sglang_cache_load_policy(
                config.clone(),
                worker_type,
                SglangCacheLoadConfig::default(),
            )
        })
        .run(
            distributed,
            EngineConfig::Dynamic {
                model: Box::new(model.build().await?),
                chat_engine_factory: None,
                prefill_load_estimator: None,
            },
        )
        .await
}

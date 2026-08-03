<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Cache/Load Worker Selection Policy

This standalone Rust crate implements an SGLang Model Gateway-style cache/load policy without changing Dynamo. It links Dynamo as a normal dependency and supplies one load scorer and one cache-aware picker. Dynamo still owns worker discovery, eligibility, scheduler accounting, validation, request dispatch, and metrics.

The scorer requests Dynamo's active-request load input. The picker requests exact effective KV overlap and follows SGLang Model Gateway's decision rule:

- Select the least-loaded worker when both absolute and relative load thresholds indicate imbalance.
- Otherwise, select the worker with the highest overlap when its match ratio exceeds the cache threshold.
- Fall back to the least-loaded worker when the cache threshold is not met.

Run the policy test from this directory:

```bash
cargo test
cargo bench --bench worker_selection
```

`src/bin/frontend.rs` links the policy into Dynamo's complete discovery-backed HTTP frontend. It adds a policy factory to `HttpFrontend::default()`; Dynamo's standard model watcher builds every decode and prefill pipeline, and only the `WorkerSelectionPolicy` factory changes.

With Dynamo workers registered at the default `dynamo.backend.generate` endpoint, run it with the same discovery environment as the workers:

```bash
DYN_DISCOVERY_BACKEND=file \
DYN_FILE_KV=/tmp/dynamo_store_kv \
cargo run --release --bin frontend -- MODEL_PATH MODEL_NAME
```

`DYN_KV_CACHE_BLOCK_SIZE` defaults to 16 and must match the workers' block size.

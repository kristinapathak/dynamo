<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM sidecar

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC `Inference` and `Control` services. It is a standalone Rust executable.

## Supported

- Aggregated generation
- NIXL prefill/decode generation
- Token and text requests through Dynamo preprocessing
- Sampling, stop conditions, structured output, logprobs, cache options, and priority
- Opaque `kv_transfer_params` handoff
- Engine identity, model, and parallelism discovery through the native `Control` service
- Optional RL worker publication for `/v1/rl/workers`

The vLLM protocol represents additional features that this sidecar does not map yet, including multimodal input, LoRA request selection, KV-aware data-parallel routing, encode workers, beam search, and `n > 1`.

## Run

Start vLLM with its released gRPC listener:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --grpc-port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on loopback or a private interface. Remote access requires network controls or a secure proxy.

Start the Dynamo worker explicitly:

```bash
dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051 \
  --model-path Qwen/Qwen3-0.6B
```

Use `VLLM_GRPC_ENDPOINT` instead of `--vllm-endpoint` when the endpoint is provided through the environment. When RL discovery is enabled with `DYN_ENABLE_RL=true`, also pass `--admin-endpoint 127.0.0.1:8001` or set `VLLM_HTTP_ENDPOINT`; the direct HTTP endpoint is required for Prime’s pause and weight-update control APIs.

RL discovery publishes internal system and admin URLs plus the registered control-route names. It is not an authenticated public admin API. Bind the system and request-plane listeners to a trusted interface and restrict access with an in-cluster network policy.

The sidecar opens eight gRPC connections by default. This avoided connection-level throttling in high-concurrency sidecar tests. Override the pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry interval, and a five-minute deadline for establishing the full connection pool and completing Control discovery. Override them with `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the corresponding `DYN_SIDECAR_GRPC_*` environment variables.

Distribution and container packaging for the executable are intentionally deferred to a follow-up change.

## Test without vLLM or a GPU

Use the CPU-only `dynamo-vllm-mocker-server` to exercise this sidecar against the same split native gRPC contract:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50051 \
  --model-path mocker-model
```

See [`../../mocker/servers/vllm/README.md`](../../mocker/servers/vllm/README.md) for aggregated and prefill/decode examples, supported Mocker configuration, and fidelity limits.

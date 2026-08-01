# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-wheel smoke tests for the public ``aisimulate predict`` path."""

from __future__ import annotations

import json
from importlib import metadata

import pytest
import yaml

from aisimulate.cli import main

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


def _run_predict(tmp_path, capsys, *, backend: str, config: dict) -> dict:
    config_path = tmp_path / f"{backend}-replay.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    exit_code = main(
        [
            "predict",
            "--stack",
            "engine",
            "--backend",
            backend,
            "--model",
            "test-model",
            "--system",
            "test-system",
            "--isl",
            "16",
            "--osl",
            "4",
            "--config",
            str(config_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    return json.loads(captured.out)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_engine_stack_runs_installed_native_runtime(tmp_path, capsys, backend):
    report = _run_predict(
        tmp_path,
        capsys,
        backend=backend,
        config={
            "request_count": 3,
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 1.0,
                "decode_ms": 0.5,
            },
        },
    )
    native_report = report["metadata"]["native_report"]
    assert {
        name: native_report[name]
        for name in (
            "num_requests",
            "completed_requests",
            "total_input_tokens",
            "total_output_tokens",
        )
    } == {
        "num_requests": 3,
        "completed_requests": 3,
        "total_input_tokens": 48,
        "total_output_tokens": 12,
    }
    assert [record["request_id"] for record in native_report["per_request"]] == [
        "synthetic-0",
        "synthetic-1",
        "synthetic-2",
    ]


def test_engine_stack_runs_disaggregated_native_runtime(tmp_path, capsys):
    role = {
        "rank": {
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 1.0,
                "decode_ms": 0.5,
            }
        }
    }
    report = _run_predict(
        tmp_path,
        capsys,
        backend="vllm",
        config={
            "deployment_mode": "disaggregated",
            "request_count": 3,
            "prefill": role,
            "decode": role,
        },
    )
    native_report = report["metadata"]["native_report"]
    assert native_report["num_requests"] == 3
    assert native_report["completed_requests"] == 3


def test_dynamo_stack_discovers_router_and_planner_provider_and_runs(tmp_path, capsys):
    adapters = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="aisimulate.adapters")
    }
    assert adapters["dynamo.router"] == "dynamo.router.simulation:create_adapter"
    assert adapters["dynamo.planner"] == "dynamo.planner.simulation:create_adapter"

    config_path = tmp_path / "dynamo-replay.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "deployment_mode": "aggregated",
                "router_mode": "kv_router",
                "planner": {
                    "enabled": True,
                    "config": {
                        "mode": "agg",
                        "optimization_target": "throughput",
                        "live_dashboard_port": 0,
                        "report_interval_hours": None,
                    },
                },
                "engine_args": {
                    "engine_type": "vllm",
                    "aic_backend": None,
                    "block_size": 4,
                    "num_gpu_blocks": 16,
                    "max_num_seqs": 4,
                    "max_num_batched_tokens": 64,
                },
                "request_count": 3,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "predict",
            "--stack",
            "dynamo",
            "--backend",
            "vllm",
            "--model",
            "test-model",
            "--system",
            "test-system",
            "--isl",
            "4",
            "--osl",
            "2",
            "--config",
            str(config_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    report = json.loads(captured.out)
    assert report["metrics"]["num_requests"] == 3.0
    assert report["metrics"]["completed_requests"] == 3.0
    assert report["metrics"]["planner_total_ticks"] == 0.0
    assert report["metadata"]["planner_total_ticks"] == 0

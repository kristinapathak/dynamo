# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``aisimulate predict`` over the canonical Spica contract."""

import builtins
import json

import pytest

import aisimulate.cli as cli
from aisimulate.cli import main
from aisimulate.runner import RunnerUnavailableError
from aisimulate.spica import ReplayReport, RunnerCapabilities, Workload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class RecordingFactory:
    def __init__(self):
        self.spec = None
        self.worker_id = None
        self.closed = False

    def capabilities(self):
        return RunnerCapabilities(
            supported_backend_topologies=(("*", "*"),),
            supported_hooks=(
                # Use the spec's concrete hooks only in tests that override this.
            ),
        )

    def create(self, worker_id):
        self.worker_id = worker_id
        return self

    def run(self, spec):
        self.spec = spec
        return ReplayReport(metrics={"z": 2.0, "a": 1.0})

    def close(self):
        self.closed = True


def _required_args(stack="engine"):
    return [
        "predict",
        "--model",
        "Qwen/Qwen3-32B",
        "--backend",
        "vllm",
        "--system",
        "h200_sxm",
        "--stack",
        stack,
    ]


def test_predict_materializes_canonical_spec_and_runner_lifecycle(capsys):
    factory = RecordingFactory()

    result = main(
        [
            *_required_args(),
            "--tp-size",
            "2",
            "--replicas",
            "3",
            "--isl",
            "128",
            "--osl",
            "16",
        ],
        runner_factory=factory,
    )

    assert result == 0
    assert factory.worker_id == 0
    assert factory.closed
    deployment = factory.spec.backend_deployment
    assert deployment.backend == "vllm"
    assert deployment.num_workers == 3
    assert deployment.agg_engine_args["aic_tp_size"] == 2
    assert factory.spec.workload == Workload(
        isl=128,
        osl=16,
        concurrency=1,
        num_request_ratio=1.0,
    ).model_dump(mode="json")
    assert not hasattr(factory.spec, "stack")
    assert json.loads(capsys.readouterr().out) == {
        "metadata": {},
        "metrics": {"a": 1.0, "z": 2.0},
    }


def test_predict_materializes_relative_trace_path(tmp_path):
    traffic = tmp_path / "traffic.yaml"
    traffic.write_text("trace:\n  path: requests.jsonl\nformat: mooncake\nspeedup: 2\n")
    factory = RecordingFactory()

    assert (
        main([*_required_args(), "--traffic", str(traffic)], runner_factory=factory)
        == 0
    )

    assert factory.spec.workload["trace_path"] == str(
        (tmp_path / "requests.jsonl").resolve()
    )
    assert factory.spec.workload["trace_format"] == "mooncake"
    assert factory.spec.workload["arrival_speedup_ratio"] == 2


def test_predict_materializes_disagg_parallel_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "deployment_mode: disagg\nnum_prefill_workers: 2\nnum_decode_workers: 4\n"
    )
    factory = RecordingFactory()

    assert (
        main(
            [
                *_required_args(),
                "--tp-size",
                "2",
                "--replicas",
                "3",
                "--isl",
                "128",
                "--osl",
                "16",
                "--config",
                str(config),
            ],
            runner_factory=factory,
        )
        == 0
    )

    deployment = factory.spec.backend_deployment
    assert deployment.parallel_config == {
        "prefill_tp": 2,
        "prefill_attention_dp": 1,
        "prefill_replicas": 2,
        "decode_tp": 2,
        "decode_attention_dp": 1,
        "decode_replicas": 4,
    }


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--isl", "128"], "--isl and --osl must be provided together"),
        (["--osl", "16"], "--isl and --osl must be provided together"),
        ([], "provide either --isl/--osl or --traffic"),
    ],
)
def test_predict_validates_traffic_shape(extra, message, capsys):
    with pytest.raises(SystemExit, match="2"):
        main([*_required_args(), *extra], runner_factory=RecordingFactory())
    assert message in capsys.readouterr().err


def test_predict_rejects_traffic_with_lengths(tmp_path, capsys):
    traffic = tmp_path / "traffic.yaml"
    traffic.write_text("trace_path: requests.jsonl\n")
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                *_required_args(),
                "--traffic",
                str(traffic),
                "--isl",
                "8",
                "--osl",
                "2",
            ],
            runner_factory=RecordingFactory(),
        )
    assert "cannot be used with --traffic" in capsys.readouterr().err


def test_predict_closes_runner_after_execution_error(capsys):
    class FailingFactory(RecordingFactory):
        def run(self, spec):
            raise ValueError("bad candidate")

    factory = FailingFactory()
    assert (
        main(
            [*_required_args(), "--isl", "8", "--osl", "2"],
            runner_factory=factory,
        )
        == 1
    )
    assert factory.closed
    assert "error: bad candidate" in capsys.readouterr().err


def test_predict_has_no_standalone_trace_flag(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(
            [*_required_args(), "--trace", "requests.jsonl"],
            runner_factory=RecordingFactory(),
        )
    assert "unrecognized arguments: --trace" in capsys.readouterr().err


def test_predict_rejects_trace_in_synthetic_workload_override(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("workload:\n  trace_path: requests.jsonl\n")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                *_required_args(),
                "--isl",
                "8",
                "--osl",
                "2",
                "--config",
                str(config),
            ],
            runner_factory=RecordingFactory(),
        )

    assert "must not set synthetic fields" in capsys.readouterr().err


def test_predict_rejects_candidate_relative_kv_load(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("workload:\n  kv_load_ratio: 0.5\n")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                *_required_args(),
                "--isl",
                "8",
                "--osl",
                "2",
                "--config",
                str(config),
            ],
            runner_factory=RecordingFactory(),
        )

    assert "resolved only by a Spica sweep" in capsys.readouterr().err


def test_predict_validates_goal_with_the_spica_contract(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("goal:\n  target: goodput\n")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                *_required_args(),
                "--isl",
                "8",
                "--osl",
                "2",
                "--config",
                str(config),
            ],
            runner_factory=RecordingFactory(),
        )

    assert "require an SLA target" in capsys.readouterr().err


def test_predict_requires_planner_enabled_to_be_boolean(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text('planner:\n  enabled: "false"\n')

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                *_required_args("dynamo"),
                "--isl",
                "8",
                "--osl",
                "2",
                "--config",
                str(config),
            ],
            runner_factory=RecordingFactory(),
        )

    assert "planner.enabled must be a boolean" in capsys.readouterr().err


def test_missing_dynamo_stack_recommends_canonical_simulate_extra(monkeypatch):
    real_import = builtins.__import__

    def missing_dynamo(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dynamo.replay.simulation":
            raise ModuleNotFoundError(
                "No module named 'dynamo.replay.simulation'",
                name="dynamo.replay.simulation",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_dynamo)

    with pytest.raises(RunnerUnavailableError, match=r"ai-dynamo\[simulate\]"):
        cli._default_runner_factory("dynamo")

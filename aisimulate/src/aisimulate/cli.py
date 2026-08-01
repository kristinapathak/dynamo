# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ``aisimulate predict`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import yaml

from .runner import NativeReplayRunnerFactory, RunnerUnavailableError
from .spica.adapter import AdapterReplaySpec, JSONValue, RuntimeHookSpec
from .spica.config import OptimizationGoal, Workload
from .spica.replay import (
    BackendDeploymentSpec,
    ReplayReport,
    ReplaySpec,
    RunnerFactory,
    canonical_json,
)

_BACKENDS = ("vllm", "sglang", "trtllm")
_STACKS = ("engine", "dynamo")


def _yaml_mapping(parser: argparse.ArgumentParser, path: Path, label: str) -> dict:
    try:
        with path.open(encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file)
    except OSError as exc:
        parser.error(f"could not read {label} {path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {label} {path}: {exc}")
    if not isinstance(payload, dict):
        parser.error(f"{label} {path} must contain a YAML mapping")
    return payload


def _resolve_traffic_paths(
    traffic: dict[str, JSONValue], source: Path
) -> dict[str, JSONValue]:
    """Resolve caller-relative trace paths before the spec crosses a process."""

    base = source.parent.resolve()

    def resolve(value: JSONValue) -> JSONValue:
        if isinstance(value, str) and value:
            path = Path(value)
            return str(path if path.is_absolute() else (base / path).resolve())
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {
                key: resolve(item) if key in {"path", "paths"} else item
                for key, item in value.items()
            }
        return value

    return {
        key: resolve(value)
        if key in {"trace", "trace_path", "trace_files", "trace_paths"}
        else value
        for key, value in traffic.items()
    }


def _predict_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "predict",
        help="evaluate one engine or Dynamo configuration",
        description="Evaluate one configuration with GPU-free replay.",
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--backend", choices=_BACKENDS, required=True)
    parser.add_argument("--system", required=True, help="Hardware system identifier")
    parser.add_argument(
        "--stack",
        choices=_STACKS,
        required=True,
        help="Select the native or Dynamo RunnerFactory",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        help="Deployment and optional Dynamo hook configuration YAML",
    )
    parser.add_argument(
        "--traffic",
        type=Path,
        help="Traffic YAML; trace inputs belong inside this file",
    )
    parser.add_argument("--isl", type=int, help="Fixed synthetic input length")
    parser.add_argument("--osl", type=int, help="Fixed synthetic output length")
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisimulate",
        description="GPU-free engine and Dynamo simulation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _predict_parser(subparsers)
    return parser


def _positive(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _role_engine_args(
    raw: object,
    *,
    role: str,
    backend: str,
    model: str,
    system: str,
    tp_size: int,
) -> dict[str, JSONValue]:
    if raw is None:
        payload: dict[str, JSONValue] = {}
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise ValueError(f"{role} engine arguments must be a mapping")
    payload.setdefault("worker_type", role)
    payload.setdefault("engine_type", backend)
    payload.setdefault("aic_backend", backend)
    payload.setdefault("aic_model_path", model)
    payload.setdefault("aic_system", system)
    payload.setdefault("aic_tp_size", tp_size)
    payload.setdefault("aic_attention_dp_size", payload.get("dp_size", 1))
    return payload


def _build_deployment(
    args: argparse.Namespace, config: dict[str, JSONValue]
) -> BackendDeploymentSpec:
    mode = str(config.pop("deployment_mode", "agg"))
    mode = {"aggregated": "agg", "disaggregated": "disagg"}.get(mode, mode)
    raw_backend_version = config.pop("backend_version", "")
    if raw_backend_version is None:
        backend_version = ""
    elif isinstance(raw_backend_version, str):
        backend_version = raw_backend_version
    else:
        raise ValueError("backend_version must be a string")
    common = {
        "deployment_mode": mode,
        "backend": args.backend,
        "backend_version": backend_version,
    }
    if mode == "agg":
        raw = config.pop("engine_args", None)
        if raw is None:
            reserved = {
                "router",
                "planner",
                "router_mode",
                "router_config",
                "planner_config",
                "workload",
                "goal",
                "num_workers",
            }
            raw = {key: config.pop(key) for key in tuple(config) if key not in reserved}
        num_workers = _positive(
            config.pop("num_workers", args.replicas),
            "num_workers",
        )
        return BackendDeploymentSpec(
            agg_engine_args=_role_engine_args(
                raw,
                role="aggregated",
                backend=args.backend,
                model=args.model,
                system=args.system,
                tp_size=args.tp_size,
            ),
            num_workers=num_workers,
            parallel_config={"tp": args.tp_size, "replicas": num_workers},
            **common,
        )
    if mode != "disagg":
        raise ValueError("deployment_mode must be agg or disagg")
    if args.backend == "trtllm":
        raise ValueError("TensorRT-LLM predict does not support disagg")
    if {"prefill_engine_args", "prefill"}.issubset(config):
        raise ValueError("cannot set both prefill_engine_args and prefill")
    if {"decode_engine_args", "decode"}.issubset(config):
        raise ValueError("cannot set both decode_engine_args and decode")
    prefill = config.pop("prefill_engine_args", None)
    if "prefill" in config:
        prefill = config.pop("prefill")
    decode = config.pop("decode_engine_args", None)
    if "decode" in config:
        decode = config.pop("decode")
    num_prefill_workers = _positive(
        config.pop("num_prefill_workers", args.replicas),
        "num_prefill_workers",
    )
    num_decode_workers = _positive(
        config.pop("num_decode_workers", args.replicas),
        "num_decode_workers",
    )
    return BackendDeploymentSpec(
        prefill_engine_args=_role_engine_args(
            prefill,
            role="prefill",
            backend=args.backend,
            model=args.model,
            system=args.system,
            tp_size=args.tp_size,
        ),
        decode_engine_args=_role_engine_args(
            decode,
            role="decode",
            backend=args.backend,
            model=args.model,
            system=args.system,
            tp_size=args.tp_size,
        ),
        num_prefill_workers=num_prefill_workers,
        num_decode_workers=num_decode_workers,
        parallel_config={
            "prefill_tp": args.tp_size,
            "prefill_attention_dp": 1,
            "prefill_replicas": num_prefill_workers,
            "decode_tp": args.tp_size,
            "decode_attention_dp": 1,
            "decode_replicas": num_decode_workers,
        },
        **common,
    )


def _dynamo_adapters(
    config: dict[str, JSONValue], deployment_mode: str
) -> dict[str, AdapterReplaySpec]:
    adapters: dict[str, AdapterReplaySpec] = {}
    router = config.pop("router", None)
    router_mode = config.pop("router_mode", None)
    router_config = config.pop("router_config", None)
    if isinstance(router, Mapping):
        router_mode = router.get("mode", router_mode or "kv_router")
        router_config = router.get("config", router_config)
    elif router is not None:
        raise ValueError("router must be a mapping")
    if router_mode in {"kv", "kv_router"} or router_config is not None:
        if router_config is None:
            router_config = {}
        if not isinstance(router_config, dict):
            raise ValueError("KV Router requires a router config mapping")
        hook = RuntimeHookSpec(
            provider="dynamo.router",
            kind="placement_policy",
            api_version=1,
            config={"router_mode": "kv_router", "router_config": router_config},
        )
        adapters["dynamo.router"] = AdapterReplaySpec(
            config={"mode": "kv_router"}, runtime_hooks=(hook,)
        )
    elif router_mode not in (None, "round_robin"):
        raise ValueError(f"unsupported router mode {router_mode!r}")

    planner = config.pop("planner", None)
    planner_config = config.pop("planner_config", None)
    enabled = planner_config is not None
    if isinstance(planner, Mapping):
        raw_enabled = planner.get("enabled", True)
        if not isinstance(raw_enabled, bool):
            raise ValueError("planner.enabled must be a boolean")
        enabled = raw_enabled
        planner_config = planner.get("config", planner_config)
    elif planner is not None:
        raise ValueError("planner must be a mapping")
    if enabled:
        if planner_config is None:
            planner_config = {"mode": deployment_mode}
        if not isinstance(planner_config, dict):
            raise ValueError("Planner config must be a mapping")
        hook = RuntimeHookSpec(
            provider="dynamo.planner",
            kind="scaling_policy",
            api_version=1,
            config={"planner_config": planner_config},
        )
        adapters["dynamo.planner"] = AdapterReplaySpec(
            config={"enabled": True}, runtime_hooks=(hook,)
        )
    return adapters


def _normalize_trace_workload(workload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    if "trace_path" not in workload:
        trace = workload.pop("trace", None)
        if isinstance(trace, str):
            workload["trace_path"] = trace
        elif isinstance(trace, Mapping):
            path = trace.get("path")
            if not isinstance(path, str):
                raise ValueError("traffic trace mapping requires a string path")
            workload["trace_path"] = path
        elif trace is not None:
            raise ValueError("traffic trace must be a path or path mapping")

    if "format" in workload:
        if "trace_format" in workload:
            raise ValueError("traffic cannot set both format and trace_format")
        workload["trace_format"] = workload.pop("format")
    if "speedup" in workload:
        if "arrival_speedup_ratio" in workload:
            raise ValueError(
                "traffic cannot set both speedup and arrival_speedup_ratio"
            )
        workload["arrival_speedup_ratio"] = workload.pop("speedup")
    return workload


def _build_predict_spec(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> ReplaySpec:
    if args.traffic is not None and (args.isl is not None or args.osl is not None):
        parser.error("--isl/--osl cannot be used with --traffic")
    if (args.isl is None) != (args.osl is None):
        parser.error("--isl and --osl must be provided together")
    if args.traffic is None and args.isl is None:
        parser.error("provide either --isl/--osl or --traffic")

    config: dict[str, JSONValue] = (
        _yaml_mapping(parser, args.config, "config") if args.config else {}
    )
    try:
        request_count = config.pop("request_count", None)
        arrival_interval_ms = config.pop("arrival_interval_ms", None)
        workload_override = config.pop("workload", {})
        if not isinstance(workload_override, dict):
            raise ValueError("config workload must be a mapping")
        if args.traffic is not None:
            workload = _normalize_trace_workload(
                _resolve_traffic_paths(
                    _yaml_mapping(parser, args.traffic, "traffic"), args.traffic
                )
            )
            workload.update(workload_override)
            concurrency = None
        else:
            workload = {
                "isl": _positive(args.isl, "isl"),
                "osl": _positive(args.osl, "osl"),
                "num_request_ratio": 1.0,
            }
            if not {
                "concurrency",
                "request_rate",
                "kv_load_ratio",
            }.intersection(workload_override):
                workload["concurrency"] = 1
            workload.update(workload_override)
            if arrival_interval_ms is not None:
                if (
                    isinstance(arrival_interval_ms, bool)
                    or not isinstance(arrival_interval_ms, (int, float))
                    or arrival_interval_ms <= 0
                ):
                    raise ValueError("arrival_interval_ms must be positive")
                workload.pop("concurrency", None)
                workload["request_rate"] = 1_000.0 / float(arrival_interval_ms)
            if request_count is not None:
                count = _positive(request_count, "request_count")
                load = workload.get("concurrency", workload.get("request_rate"))
                if isinstance(load, bool) or not isinstance(load, (int, float)):
                    raise ValueError("request_count requires concrete replay load")
                workload["num_request_ratio"] = count / float(load)
            concurrency_value = workload.get("concurrency")
            concurrency = (
                _positive(concurrency_value, "concurrency")
                if concurrency_value is not None
                else None
            )
        trace_block_size = workload.pop("trace_block_size", None)
        validated_workload = Workload.model_validate(workload)
        if validated_workload.kv_load_ratio is not None:
            raise ValueError(
                "aisimulate predict requires concrete concurrency or request_rate; "
                "kv_load_ratio is resolved only by a Spica sweep"
            )
        workload = validated_workload.model_dump(mode="json")
        if trace_block_size is not None:
            workload["trace_block_size"] = _positive(
                trace_block_size, "trace_block_size"
            )
        goal = config.pop("goal", {"target": "throughput"})
        if not isinstance(goal, dict):
            raise ValueError("config goal must be a mapping")
        goal = OptimizationGoal.model_validate(goal).model_dump(mode="json")
        deployment = _build_deployment(args, config)
        adapters = (
            _dynamo_adapters(config, deployment.deployment_mode)
            if args.stack == "dynamo"
            else {}
        )
        if config:
            raise ValueError("unrecognized config fields: " + ", ".join(sorted(config)))
        spec = ReplaySpec(
            backend_deployment=deployment,
            workload=workload,
            goal=goal,
            concurrency=concurrency,
            adapters=adapters,
        )
        canonical_json(spec)
        return spec
    except (TypeError, ValueError) as exc:
        parser.error(f"invalid replay configuration: {exc}")


def _default_runner_factory(stack: str) -> RunnerFactory:
    if stack == "engine":
        return NativeReplayRunnerFactory(include_native_report=True)
    try:
        from dynamo.replay.simulation import DynamoReplayRunnerFactory
    except ModuleNotFoundError as exc:
        if exc.name not in {"dynamo", "dynamo.replay", "dynamo.replay.simulation"}:
            raise RunnerUnavailableError(
                "Dynamo simulation support could not be loaded because runtime "
                f"dependency {exc.name!r} is missing: {exc}"
            ) from exc
        raise RunnerUnavailableError(
            "Dynamo simulation support is not installed.\n"
            "Install it with: python -m pip install 'ai-dynamo[simulate]'"
        ) from exc
    except ImportError as exc:
        raise RunnerUnavailableError(
            f"Dynamo simulation support is installed but could not be loaded: {exc}"
        ) from exc
    return DynamoReplayRunnerFactory()


def _run(factory: RunnerFactory, spec: ReplaySpec) -> ReplayReport:
    factory.capabilities().require_compatible(spec)
    runner = factory.create(worker_id=0)
    try:
        report = runner.run(spec)
        if not isinstance(report, ReplayReport):
            raise TypeError("runner.run must return ReplayReport")
        return report
    finally:
        runner.close()


def _canonical_report(report: ReplayReport) -> str:
    return json.dumps(
        asdict(report),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: RunnerFactory | None = None,
) -> int:
    """Run the public CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "predict":
        parser.error(f"unknown command: {args.command}")
    spec = _build_predict_spec(parser, args)
    try:
        factory = runner_factory or _default_runner_factory(args.stack)
        report = _run(factory, spec)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_canonical_report(report))
    return 0

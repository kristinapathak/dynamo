# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

import inspect
import logging

from gpu_memory_service.integrations.sglang.memory_saver import (
    get_gms_memory_saver_impl,
)

logger = logging.getLogger(__name__)

_model_runner_patched = False
_static_state_patched = False


def patch_model_runner() -> None:
    """Patch SGLang's ModelRunner to size KV cache with GMS-resident weights.

    SGLang's KV sizing formula reserves dynamic headroom from a free-memory
    snapshot taken before its own model load. In GMS read mode, the committed
    weight handles already exist in the GMS server before that snapshot, so the
    snapshot is lower by those weights. Add just those preloaded weight bytes
    back to the baseline. Do not adjust write mode: weights are loaded after
    the snapshot there, so upstream's formula already subtracts them correctly.
    """
    global _model_runner_patched

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_init_memory_pool = ModelRunner.init_memory_pool
    memory_arg_name = next(
        (
            name
            for name in inspect.signature(original_init_memory_pool).parameters
            if name != "self"
        ),
        None,
    )

    def patched_init_memory_pool(self, *args, **kwargs):
        """Patch memory baseline for SGLang old/new init_memory_pool signatures."""
        impl = get_gms_memory_saver_impl()
        preloaded_weights_gib = 0.0
        if impl is not None:
            preloaded_weights_gib = impl.preloaded_weights_bytes / (1 << 30)

        if preloaded_weights_gib > 0 and memory_arg_name in (
            "pre_model_load_memory",
            "total_gpu_memory",
        ):
            if args:
                old_value = args[0]
                new_value = (
                    old_value + preloaded_weights_gib
                    if isinstance(old_value, (int, float))
                    else old_value
                )
                args = (new_value,) + args[1:]
            elif memory_arg_name in kwargs:
                old_value = kwargs[memory_arg_name]
                new_value = (
                    old_value + preloaded_weights_gib
                    if isinstance(old_value, (int, float))
                    else old_value
                )
                kwargs = dict(kwargs)
                kwargs[memory_arg_name] = new_value
            else:
                old_value = None
                new_value = None

            if isinstance(old_value, (int, float)) and isinstance(
                new_value, (int, float)
            ):
                logger.info(
                    "[GMS] Adjusted %s for preloaded weights: "
                    "%.2f GiB + %.2f GiB = %.2f GiB",
                    memory_arg_name,
                    old_value,
                    preloaded_weights_gib,
                    new_value,
                )
            else:
                logger.info(
                    "[GMS] Could not adjust %s for preloaded weights; value=%r",
                    memory_arg_name,
                    old_value,
                )
        elif impl is not None and impl.imported_weights_bytes > 0:
            if preloaded_weights_gib > 0:
                logger.info(
                    "[GMS] Leaving %s unchanged; unsupported SGLang "
                    "init_memory_pool signature for preloaded weights",
                    memory_arg_name,
                )
            else:
                logger.info(
                    "[GMS] Leaving %s unchanged; weights were loaded by this process",
                    memory_arg_name,
                )

        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = patched_init_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.init_memory_pool")


def patch_static_state_for_gms() -> None:
    """No-op SGLang's _export/_import_static_state when using GMS.

    SGLang's release_memory_occupation clones every named buffer via
    buffer.detach().clone() through the default CUDA allocator, then restores
    them during resume_memory_occupation.
    This patch must run inside the scheduler child process (which uses
    multiprocessing spawn).  It is triggered by the GMSModelLoader import
    in model_loader.py, which executes at module level in the child.
    """
    import os

    global _static_state_patched
    logger.info(
        "[GMS] patch_static_state_for_gms called (pid=%d, already_patched=%s)",
        os.getpid(),
        _static_state_patched,
    )
    if _static_state_patched:
        return

    try:
        from sglang.srt.managers import scheduler_update_weights_mixin as _mixin

        def _export_noop(model):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            return dict(buffers=[])

        def _import_noop(model, static_params):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            pass

        _mixin._export_static_state = _export_noop
        _mixin._import_static_state = _import_noop
        _static_state_patched = True
        logger.info(
            "[GMS] Patched _export/_import_static_state -> no-op (pid=%d)",
            os.getpid(),
        )
    except Exception:
        logger.warning(
            "[GMS] Could not patch scheduler_update_weights_mixin: ",
            exc_info=True,
        )

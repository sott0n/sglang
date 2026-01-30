# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tenstorrent backend utilities for SGLang."""

import functools
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import get_tt_memory_capacity, is_tt

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Global mesh device reference
_mesh_device = None


def is_tt_available() -> bool:
    """Check if Tenstorrent device is available."""
    return is_tt()


def _call_once(fn):
    """Decorator to ensure a function is only called once."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(fn, "_has_been_called", False):
            logger.debug("Function %s has already been called.", fn.__name__)
            return
        fn._has_been_called = True
        return fn(*args, **kwargs)

    return wrapper


def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for Tenstorrent backend.

    Args:
        args: ServerArgs instance to configure
    """
    # TT uses torch_native attention backend for now
    # In the future, this can be extended to use TT-specific attention
    if args.attention_backend is None:
        args.attention_backend = "torch_native"
    if args.prefill_attention_backend is None:
        args.prefill_attention_backend = "torch_native"
    if args.decode_attention_backend is None:
        args.decode_attention_backend = "torch_native"

    # Set default page size if not specified
    if args.page_size is None:
        args.page_size = 16

    # TT memory settings
    tt_mem = get_tt_memory_capacity()
    if tt_mem is not None:
        # Set chunked prefill size based on available memory
        if args.chunked_prefill_size is None:
            if tt_mem <= 12 * 1024:  # 12GB or less (Wormhole)
                args.chunked_prefill_size = 2 * 1024
            else:
                args.chunked_prefill_size = 4 * 1024

        # Set CUDA graph max batch size
        if args.cuda_graph_max_bs is None:
            if tt_mem <= 12 * 1024:
                args.cuda_graph_max_bs = 8
            else:
                args.cuda_graph_max_bs = 16

    # TT does not support CustomAllReduce
    args.disable_custom_all_reduce = True

    # Disable CUDA graphs for TT (use TT trace mode instead)
    if args.disable_cuda_graph is None:
        args.disable_cuda_graph = True

    # Set sampling backend to pytorch for TT
    if args.sampling_backend is None:
        args.sampling_backend = "pytorch"


@_call_once
def init_tt_backend():
    """
    Initialize Tenstorrent backend. This function should be called only once.
    """
    global _mesh_device

    if not is_tt():
        raise RuntimeError("TT backend initialization called on non-TT device.")

    try:
        import ttnn

        logger.info("Initializing Tenstorrent backend...")

        # Get available device IDs
        device_ids = ttnn.get_device_ids()
        if not device_ids:
            raise RuntimeError("No Tenstorrent devices found.")

        logger.info(f"Found {len(device_ids)} Tenstorrent device(s): {device_ids}")

        # For single device setup, open a single device
        # For multi-device, use mesh device
        if len(device_ids) == 1:
            _mesh_device = ttnn.open_device(device_id=device_ids[0])
        else:
            # Create mesh device for multi-chip configurations
            mesh_shape = ttnn.MeshShape(1, len(device_ids))
            _mesh_device = ttnn.open_mesh_device(
                mesh_shape=mesh_shape,
                dispatch_core_type=ttnn.DispatchCoreType.WORKER,
            )

        # Enable program cache for better performance
        if hasattr(_mesh_device, "enable_program_cache"):
            _mesh_device.enable_program_cache()

        logger.info("Tenstorrent backend initialized successfully.")

    except ImportError as e:
        raise ImportError(
            "ttnn is required for Tenstorrent backend. "
            "Please install tt-metal: https://github.com/tenstorrent/tt-metal"
        ) from e
    except Exception as e:
        logger.error(f"Failed to initialize Tenstorrent backend: {e}")
        raise


def get_mesh_device():
    """Get the global mesh device instance."""
    global _mesh_device
    if _mesh_device is None:
        init_tt_backend()
    return _mesh_device


def close_tt_backend():
    """Close the Tenstorrent backend and release resources."""
    global _mesh_device

    if _mesh_device is not None:
        try:
            import ttnn

            if hasattr(_mesh_device, "close"):
                # Single device
                ttnn.close_device(_mesh_device)
            else:
                # Mesh device
                ttnn.close_mesh_device(_mesh_device)
            _mesh_device = None
            logger.info("Tenstorrent backend closed successfully.")
        except Exception as e:
            logger.warning(f"Error closing Tenstorrent backend: {e}")


def tt_synchronize():
    """Synchronize Tenstorrent device operations."""
    device = get_mesh_device()
    if device is not None:
        try:
            import ttnn

            ttnn.synchronize_device(device)
        except Exception:
            pass


def get_tt_device_count() -> int:
    """Get the number of available Tenstorrent devices."""
    try:
        import ttnn

        return len(ttnn.get_device_ids())
    except Exception:
        return 0

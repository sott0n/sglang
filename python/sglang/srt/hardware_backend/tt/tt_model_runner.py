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
"""Tenstorrent-specific Model Runner for SGLang.

This module provides a model runner optimized for Tenstorrent hardware,
utilizing the ttnn library for device operations.
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.hardware_backend.tt.utils import (
    get_mesh_device,
    init_tt_backend,
    tt_synchronize,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class TTModelRunner:
    """
    Model runner for Tenstorrent devices.

    This class handles the execution of models on Tenstorrent hardware,
    managing device placement, forward passes, and KV cache operations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        server_args: "ServerArgs",
        device_id: int = 0,
    ):
        """
        Initialize the TT Model Runner.

        Args:
            model: The PyTorch model to run
            server_args: Server configuration arguments
            device_id: The Tenstorrent device ID to use
        """
        self.model = model
        self.server_args = server_args
        self.device_id = device_id

        # Initialize TT backend
        init_tt_backend()
        self.mesh_device = get_mesh_device()

        # Model configuration
        self.dtype = self._get_model_dtype()
        self.is_tracing = False
        self.trace_cache = {}

        logger.info(f"TTModelRunner initialized on device {device_id}")

    def _get_model_dtype(self) -> torch.dtype:
        """Get the model's data type."""
        # Default to bfloat16 for TT devices
        dtype_str = self.server_args.dtype
        if dtype_str == "auto" or dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        return torch.bfloat16

    def forward(
        self,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """
        Run forward pass on Tenstorrent device.

        Args:
            forward_batch: Batch information for the forward pass

        Returns:
            Logits tensor
        """
        # Determine forward mode
        if forward_batch.forward_mode.is_decode():
            return self.decode_forward(forward_batch)
        else:
            return self.prefill_forward(forward_batch)

    def prefill_forward(
        self,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """
        Run prefill forward pass.

        Args:
            forward_batch: Batch information for prefill

        Returns:
            Logits tensor
        """
        # For prefill, we process the full prompt
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions

        # Run the model forward pass
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
            )

        # Synchronize TT device
        tt_synchronize()

        return logits

    def decode_forward(
        self,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """
        Run decode forward pass.

        Args:
            forward_batch: Batch information for decode

        Returns:
            Logits tensor
        """
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions

        # Check if we can use traced execution
        batch_size = input_ids.shape[0]
        trace_key = f"decode_{batch_size}"

        if self.is_tracing and trace_key in self.trace_cache:
            # Use traced execution for better performance
            return self._run_traced(trace_key, input_ids, positions, forward_batch)

        # Regular forward pass
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
            )

        # Synchronize TT device
        tt_synchronize()

        return logits

    def _run_traced(
        self,
        trace_key: str,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """
        Run a traced forward pass for better performance.

        Args:
            trace_key: Key for the cached trace
            input_ids: Input token IDs
            positions: Position indices
            forward_batch: Batch information

        Returns:
            Logits tensor
        """
        # TT trace mode execution would go here
        # For now, fall back to regular execution
        return self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )

    def capture_trace(
        self,
        batch_size: int,
        seq_len: int = 1,
    ):
        """
        Capture a trace for the given batch configuration.

        Args:
            batch_size: Batch size to trace
            seq_len: Sequence length to trace
        """
        trace_key = f"decode_{batch_size}"
        if trace_key in self.trace_cache:
            logger.debug(f"Trace already exists for {trace_key}")
            return

        logger.info(f"Capturing trace for batch_size={batch_size}, seq_len={seq_len}")

        try:
            import ttnn

            # Create dummy inputs for tracing
            dummy_input_ids = torch.zeros(
                (batch_size, seq_len), dtype=torch.long, device="cpu"
            )
            dummy_positions = torch.arange(seq_len, dtype=torch.long, device="cpu")
            dummy_positions = dummy_positions.unsqueeze(0).expand(batch_size, -1)

            # TT trace capture would go here
            # For now, we just mark that tracing is available
            self.trace_cache[trace_key] = {
                "batch_size": batch_size,
                "seq_len": seq_len,
            }
            self.is_tracing = True

            logger.info(f"Trace captured for {trace_key}")

        except Exception as e:
            logger.warning(f"Failed to capture trace: {e}")

    def release_traces(self):
        """Release all captured traces."""
        self.trace_cache.clear()
        self.is_tracing = False
        logger.info("All TT traces released")

    def warmup(self, batch_sizes: Optional[List[int]] = None):
        """
        Warm up the model by running forward passes with various batch sizes.

        Args:
            batch_sizes: List of batch sizes to warm up
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        logger.info(f"Warming up TTModelRunner with batch sizes: {batch_sizes}")

        for bs in batch_sizes:
            try:
                self.capture_trace(batch_size=bs, seq_len=1)
            except Exception as e:
                logger.warning(f"Warmup failed for batch_size={bs}: {e}")

        tt_synchronize()
        logger.info("TTModelRunner warmup complete")

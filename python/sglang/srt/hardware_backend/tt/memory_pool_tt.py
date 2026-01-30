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
"""Tenstorrent-specific memory pool implementation for KV cache management."""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


class TTMHATokenToKVPool(MHATokenToKVPool):
    """
    Multi-Head Attention KV cache pool for Tenstorrent devices.

    This class manages KV cache memory allocation and access patterns
    optimized for Tenstorrent hardware.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = False,  # TT doesn't use CUDA streams
        enable_kv_cache_copy: bool = False,
    ):
        """
        Initialize the TT MHA KV cache pool.

        Args:
            size: Total number of tokens to allocate
            page_size: Size of each page
            dtype: Data type for KV cache
            head_num: Number of attention heads
            head_dim: Dimension of each head
            layer_num: Number of transformer layers
            device: Device string
            enable_memory_saver: Whether to enable memory saver
            start_layer: Starting layer index
            end_layer: Ending layer index
            enable_alt_stream: Not used for TT
            enable_kv_cache_copy: Whether to enable KV cache copying
        """
        # TT doesn't use CUDA streams, disable alt stream
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=False,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        """Create KV cache buffers for TT device."""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # Create KV buffers in a layout suitable for TT
            # [layer_num, 2, size/page_size + 1, page_size, head_num, head_dim]
            # The padded slot 0 is used for writing dummy outputs from padded tokens.

            self.kv_buffer = torch.zeros(
                (
                    2,  # K and V
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """
        Set the KV cache buffer for a given layer.

        Args:
            layer: The attention layer
            loc: Location indices for the cache
            cache_k: Key cache tensor
            cache_v: Value cache tensor
            k_scale: Optional scale for keys
            v_scale: Optional scale for values
            layer_id_override: Optional override for layer ID
        """
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        # Apply scaling if needed
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k / k_scale
            if v_scale is not None:
                cache_v = cache_v / v_scale
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        # Use index_copy for TT-compatible scatter operation
        loc = loc.to(torch.int64)
        k_buffer = self.k_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )
        v_buffer = self.v_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )

        k_buffer.index_copy_(0, loc, cache_k.view(-1, self.head_num, self.head_dim))
        v_buffer.index_copy_(0, loc, cache_v.view(-1, self.head_num, self.head_dim))

    def get_contiguous_buf_infos(self):
        """Get contiguous buffer information for disaggregated execution."""
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens


class TTMLATokenToKVPool(MLATokenToKVPool):
    """
    Multi-Latent Attention KV cache pool for Tenstorrent devices.

    This class manages KV cache memory for MLA-style attention mechanisms
    optimized for Tenstorrent hardware.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """
        Initialize the TT MLA KV cache pool.

        Args:
            size: Total number of tokens to allocate
            page_size: Size of each page
            dtype: Data type for KV cache
            kv_lora_rank: Rank of KV LoRA
            qk_rope_head_dim: Dimension of QK RoPE head
            index_head_dim: Dimension of index head
            layer_num: Number of transformer layers
            device: Device string
            enable_memory_saver: Whether to enable memory saver
            start_layer: Starting layer index
            end_layer: Ending layer index
        """
        # Call grandparent init to avoid parent's buffer creation
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # Create K buffer for compressed KV
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            # Create V buffer for RoPE keys
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            # Create index K buffer if needed
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        """Get total size of KV cache in bytes."""
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        """Get KV buffer for a specific layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_key_buffer(self, layer_id: int):
        """Get key buffer for a specific layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        """Get value buffer for a specific layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        """Get index key buffer for a specific layer."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer]

    def get_contiguous_buf_infos(self):
        """Get contiguous buffer information for disaggregated execution."""
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """
        Set the KV cache buffer for a given layer.

        Args:
            layer: The attention layer
            loc: Location indices for the cache
            cache_k: Key cache tensor (compressed KV)
            cache_v: Value cache tensor (RoPE keys)
        """
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        # Use index_copy for TT-compatible scatter operation
        loc = loc.to(torch.int64)
        k_target = self.k_buffer[layer_id - self.start_layer].view(
            -1, 1, self.kv_lora_rank
        )
        v_target = self.v_buffer[layer_id - self.start_layer].view(
            -1, 1, self.qk_rope_head_dim
        )

        k_target.index_copy_(0, loc, cache_k.view(-1, 1, self.kv_lora_rank))
        v_target.index_copy_(0, loc, cache_v.view(-1, 1, self.qk_rope_head_dim))

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        """
        Set the index key buffer for a given layer.

        Args:
            layer_id: Layer identifier
            loc: Location indices for the cache
            index_k: Index key tensor
        """
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        loc = loc.to(torch.int64)
        index_k_target = self.index_k_buffer[layer_id - self.start_layer].view(
            -1, 1, self.index_head_dim
        )
        index_k_target.index_copy_(0, loc, index_k.view(-1, 1, self.index_head_dim))

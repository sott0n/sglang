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
"""Tenstorrent-specific model loader for SGLang.

This module provides model loading functionality optimized for Tenstorrent hardware,
utilizing the tt-metal-models library for optimized model implementations.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import BaseModelLoader, DefaultModelLoader
from sglang.srt.model_loader.utils import get_model_architecture, set_default_torch_dtype

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

# Mapping from standard model names to TT-optimized model names
TT_MODEL_MAPPING: Dict[str, str] = {
    "LlamaForCausalLM": "TTLlamaForCausalLM",
    "MistralForCausalLM": "TTMistralForCausalLM",
    "Qwen2ForCausalLM": "TTQwen2ForCausalLM",
}


def get_tt_model_class(model_name: str) -> Optional[str]:
    """
    Get the TT-optimized model class name for a given standard model name.

    Args:
        model_name: Standard model class name (e.g., "LlamaForCausalLM")

    Returns:
        TT-optimized model class name if available, None otherwise
    """
    return TT_MODEL_MAPPING.get(model_name)


def try_import_tt_model(model_name: str) -> Optional[Type[nn.Module]]:
    """
    Try to import a TT-optimized model class.

    Args:
        model_name: TT model class name (e.g., "TTLlamaForCausalLM")

    Returns:
        Model class if available, None otherwise
    """
    try:
        # Try importing from tt-metal-models
        import tt_metal_models

        if hasattr(tt_metal_models, model_name):
            return getattr(tt_metal_models, model_name)
    except ImportError:
        pass

    try:
        # Try importing from models.tt submodule
        from tt_metal_models import models

        if hasattr(models, model_name):
            return getattr(models, model_name)
    except ImportError:
        pass

    return None


class TTModelLoader(BaseModelLoader):
    """
    Model loader for Tenstorrent devices.

    This loader attempts to load TT-optimized model implementations from
    tt-metal-models. If a TT-optimized version is not available, it falls
    back to the default model loader.
    """

    def __init__(self, load_config: LoadConfig):
        """
        Initialize the TT model loader.

        Args:
            load_config: Load configuration
        """
        super().__init__(load_config)
        self._default_loader = DefaultModelLoader(load_config)
        self._tt_models_available = self._check_tt_models_available()

    def _check_tt_models_available(self) -> bool:
        """Check if tt-metal-models is available."""
        try:
            import tt_metal_models  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "tt-metal-models not available. "
                "TT-optimized models will not be used. "
                "Install from: https://github.com/tenstorrent/tt-metal"
            )
            return False

    def download_model(self, model_config: "ModelConfig") -> None:
        """
        Download the model weights.

        Args:
            model_config: Model configuration
        """
        self._default_loader.download_model(model_config)

    def load_model(
        self,
        *,
        model_config: "ModelConfig",
        device_config: "DeviceConfig",
    ) -> nn.Module:
        """
        Load the model for Tenstorrent device.

        This method attempts to load a TT-optimized model implementation.
        If not available, it falls back to the standard model with a warning.

        Args:
            model_config: Model configuration
            device_config: Device configuration

        Returns:
            Loaded model
        """
        # Get the standard model architecture
        model_class, arch = get_model_architecture(model_config)
        standard_model_name = model_class.__name__

        # Check if TT-optimized version is available
        tt_model_name = get_tt_model_class(standard_model_name)
        tt_model_class = None

        if tt_model_name and self._tt_models_available:
            tt_model_class = try_import_tt_model(tt_model_name)
            if tt_model_class:
                logger.info(
                    f"Using TT-optimized model: {tt_model_name} "
                    f"(original: {standard_model_name})"
                )

        if tt_model_class is None:
            logger.info(
                f"TT-optimized model not available for {standard_model_name}. "
                "Using standard model implementation."
            )
            # Fall back to default loader
            return self._default_loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )

        # Load TT-optimized model
        return self._load_tt_model(
            model_class=tt_model_class,
            model_config=model_config,
            device_config=device_config,
        )

    def _load_tt_model(
        self,
        model_class: Type[nn.Module],
        model_config: "ModelConfig",
        device_config: "DeviceConfig",
    ) -> nn.Module:
        """
        Load a TT-optimized model.

        Args:
            model_class: TT model class
            model_config: Model configuration
            device_config: Device configuration

        Returns:
            Loaded TT model
        """
        from sglang.srt.hardware_backend.tt.utils import get_mesh_device, init_tt_backend

        # Initialize TT backend
        init_tt_backend()
        mesh_device = get_mesh_device()

        with set_default_torch_dtype(model_config.dtype):
            # Initialize the TT model
            # TT models typically accept mesh_device as a parameter
            try:
                model = model_class(
                    config=model_config.hf_config,
                    mesh_device=mesh_device,
                )
            except TypeError:
                # If mesh_device is not accepted, try without it
                model = model_class(config=model_config.hf_config)

            # Load weights using the default weight loading mechanism
            self._load_weights(model, model_config)

        return model

    def _load_weights(
        self,
        model: nn.Module,
        model_config: "ModelConfig",
    ) -> None:
        """
        Load weights into the TT model.

        Args:
            model: Model to load weights into
            model_config: Model configuration
        """
        # Use default loader's weight loading mechanism
        source = DefaultModelLoader.Source.init_new(model_config, model)

        for name, loaded_weight in self._default_loader._get_weights_iterator(source):
            # Try to load the weight into the model
            try:
                param = model
                for attr in name.split("."):
                    if attr.isdigit():
                        param = param[int(attr)]
                    else:
                        param = getattr(param, attr, None)
                        if param is None:
                            break

                if param is not None and hasattr(param, "data"):
                    param.data.copy_(loaded_weight)
            except Exception as e:
                logger.debug(f"Could not load weight {name}: {e}")

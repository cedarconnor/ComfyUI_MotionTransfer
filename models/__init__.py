"""Unified model management for RAFT and SEA-RAFT optical flow models.

This module provides a clean interface for loading both RAFT (2020) and SEA-RAFT (2024)
models with automatic caching and device management.

Example:
    >>> from models import OpticalFlowModel
    >>> model, model_type = OpticalFlowModel.load('sea-raft-medium', device)
    >>> flow_low, flow_up = model(img1, img2, iters=8, test_mode=True)
"""
import torch
from typing import Tuple, Union

from .raft_loader import RAFTLoader, RAFT_Original
from .searaft_loader import SEARAFTLoader, SEARAFT


class OpticalFlowModel:
    """Unified interface for RAFT and SEA-RAFT models.

    This class automatically determines which model loader to use based on
    the model name and provides a consistent interface for both model types.
    """

    @staticmethod
    def load(model_name: str, device: torch.device) -> Tuple[Union[RAFT_Original, SEARAFT], str]:
        """Load optical flow model with automatic type detection.

        Args:
            model_name: Model identifier. Options:
                - RAFT models: 'raft-things', 'raft-sintel', 'raft-small'
                - SEA-RAFT models: 'sea-raft-small', 'sea-raft-medium', 'sea-raft-large'
            device: torch.device to load model on (e.g., torch.device('cuda'))

        Returns:
            Tuple of (model, model_type) where:
                - model: Loaded PyTorch model in eval mode
                - model_type: String 'raft' or 'searaft' indicating model type

        Raises:
            ValueError: If model_name is not recognized
            FileNotFoundError: If RAFT checkpoint file not found
            ImportError: If SEA-RAFT dependencies missing
            RuntimeError: If model loading fails

        Example:
            >>> device = torch.device('cuda')
            >>> model, model_type = OpticalFlowModel.load('sea-raft-medium', device)
            >>> print(f"Loaded {model_type} model")
            Loaded searaft model
        """
        if model_name.startswith("sea-raft"):
            # Load SEA-RAFT model
            model = SEARAFTLoader.load(model_name, device)
            return model, 'searaft'
        elif model_name.startswith("raft"):
            # Load original RAFT model
            model = RAFTLoader.load(model_name, device)
            return model, 'raft'
        else:
            raise ValueError(
                f"Unknown model type: {model_name}\n"
                f"Model name must start with 'raft' or 'sea-raft'"
            )

    @staticmethod
    def get_recommended_iters(model_type: str) -> int:
        """Get recommended iteration count for model type.

        SEA-RAFT converges faster and needs fewer iterations than original RAFT
        for the same quality level.

        Args:
            model_type: Either 'raft' or 'searaft'

        Returns:
            Recommended number of refinement iterations

        Example:
            >>> iters = OpticalFlowModel.get_recommended_iters('searaft')
            >>> print(iters)
            8
        """
        if model_type == 'searaft':
            return 8  # SEA-RAFT: faster convergence
        else:
            return 12  # RAFT: needs more iterations

    @staticmethod
    def get_available_models():
        """Get list of all available model names.

        Returns:
            Dictionary with 'raft' and 'searaft' model lists

        Example:
            >>> models = OpticalFlowModel.get_available_models()
            >>> print(models['searaft'])
            ['sea-raft-small', 'sea-raft-medium', 'sea-raft-large']
        """
        return {
            'raft': ['raft-things', 'raft-sintel', 'raft-small'],
            'searaft': ['sea-raft-small', 'sea-raft-medium', 'sea-raft-large']
        }

    @staticmethod
    def clear_cache():
        """Clear all cached models to free memory.

        Useful when switching between many different models or when
        VRAM is limited.

        Example:
            >>> OpticalFlowModel.clear_cache()
            [RAFT Loader] Cache cleared
            [SEA-RAFT Loader] Cache cleared
        """
        RAFTLoader.clear_cache()
        SEARAFTLoader.clear_cache()


# Export public API
__all__ = [
    'OpticalFlowModel',
    'RAFTLoader',
    'SEARAFTLoader',
    'RAFT_Original',
    'SEARAFT'
]

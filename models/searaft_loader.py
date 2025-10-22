"""SEA-RAFT model loader - uses bundled vendor code + HuggingFace Hub.

This module handles loading of SEA-RAFT models (ECCV 2024) using the bundled
searaft_vendor code and automatic model downloads from HuggingFace Hub.
"""
import torch
import argparse
import os
from typing import Optional

# Use relative import from vendor code
from ..searaft_vendor.core.raft import RAFT as SEARAFT


class SEARAFTLoader:
    """Handles SEA-RAFT model loading with HuggingFace Hub integration."""

    _cache = {}  # Model cache: {(model_name, device): model}

    @classmethod
    def load(cls, model_name: str, device: torch.device) -> SEARAFT:
        """Load SEA-RAFT model from HuggingFace Hub.

        Args:
            model_name: 'sea-raft-small', 'sea-raft-medium', or 'sea-raft-large'
            device: torch.device to load model on

        Returns:
            Loaded SEA-RAFT model in eval mode

        Raises:
            ValueError: If model_name is unknown
            ImportError: If huggingface-hub is not installed
            RuntimeError: If model download or loading fails
        """
        cache_key = (model_name, str(device))

        # Return cached model if available
        if cache_key in cls._cache:
            print(f"[SEA-RAFT Loader] Using cached model: {model_name}")
            return cls._cache[cache_key]

        # Check for huggingface-hub dependency
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "SEA-RAFT requires huggingface-hub for automatic model downloads.\n\n"
                "Install with:\n"
                "  pip install huggingface-hub>=0.20.0\n\n"
                "Alternatively, use original RAFT models (raft-sintel, raft-things, raft-small) "
                "which don't require HuggingFace Hub."
            )

        # Map model names to HuggingFace repository IDs
        hf_models = {
            "sea-raft-small": "MemorySlices/SEA-RAFT-S",
            "sea-raft-medium": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
            "sea-raft-large": "MemorySlices/SEA-RAFT-L",
        }

        if model_name not in hf_models:
            raise ValueError(
                f"Unknown SEA-RAFT model: {model_name}\n"
                f"Available models: {list(hf_models.keys())}"
            )

        repo_id = hf_models[model_name]
        print(f"[SEA-RAFT Loader] Loading from HuggingFace: {repo_id}")
        print("[SEA-RAFT Loader] First run downloads model (~100-200MB), subsequent runs use cache...")

        try:
            # Download checkpoint from HuggingFace Hub (auto-caches to ~/.cache/huggingface)
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.pth",
                cache_dir=None  # Uses default cache directory
            )
            print(f"[SEA-RAFT Loader] Downloaded to: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Create SEA-RAFT model with appropriate architecture
            args = argparse.Namespace()
            args.small = (model_name == "sea-raft-small")
            args.mixed_precision = False
            args.alternate_corr = False

            model = SEARAFT(args)
            model.load_state_dict(checkpoint)
            model = model.to(device).eval()

            # Cache the model
            cls._cache[cache_key] = model
            print(f"[SEA-RAFT Loader] ✓ Successfully loaded {model_name} on {device}")

            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SEA-RAFT model from HuggingFace.\n\n"
                f"Model: {repo_id}\n"
                f"Error: {e}\n\n"
                f"Troubleshooting:\n"
                f"1. Check your internet connection\n"
                f"2. Verify PyTorch >= 2.2.0: pip install --upgrade torch\n"
                f"3. Update huggingface-hub: pip install --upgrade huggingface-hub\n"
                f"4. Try using RAFT models instead (raft-sintel, raft-things)\n\n"
                f"If the issue persists, report at:\n"
                f"https://github.com/cedarconnor/ComfyUI_MotionTransfer/issues"
            )

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._cache.clear()
        print("[SEA-RAFT Loader] Cache cleared")


# Export public API
__all__ = ['SEARAFTLoader', 'SEARAFT']

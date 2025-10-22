"""RAFT model loader - uses bundled vendor code.

This module handles loading of original RAFT models (2020) using the bundled
raft_vendor code. No external dependencies required.
"""
import torch
import argparse
import os
from collections import OrderedDict

# Use relative import from vendor code (no sys.path hacking!)
from ..raft_vendor.core.raft import RAFT as RAFT_Original


class RAFTLoader:
    """Handles RAFT model loading and caching."""

    _cache = {}  # Model cache: {(model_name, device): model}

    @classmethod
    def load(cls, model_name: str, device: torch.device) -> RAFT_Original:
        """Load RAFT model from checkpoint.

        Args:
            model_name: 'raft-things', 'raft-sintel', or 'raft-small'
            device: torch.device to load model on

        Returns:
            Loaded RAFT model in eval mode

        Raises:
            ValueError: If model_name is unknown
            FileNotFoundError: If checkpoint file not found
        """
        cache_key = (model_name, str(device))

        # Return cached model if available
        if cache_key in cls._cache:
            print(f"[RAFT Loader] Using cached model: {model_name}")
            return cls._cache[cache_key]

        # Determine checkpoint filename
        checkpoint_map = {
            "raft-things": "raft-things.pth",
            "raft-sintel": "raft-sintel.pth",
            "raft-small": "raft-small.pth",
        }

        if model_name not in checkpoint_map:
            raise ValueError(
                f"Unknown RAFT model: {model_name}\n"
                f"Available models: {list(checkpoint_map.keys())}"
            )

        # Find ComfyUI base directory
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        comfy_base = os.path.abspath(os.path.join(package_dir, '..', '..'))
        checkpoint_path = os.path.join(comfy_base, "models", "raft", checkpoint_map[model_name])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"RAFT checkpoint not found: {checkpoint_path}\n\n"
                f"Download RAFT models from: https://github.com/princeton-vl/RAFT#demos\n"
                f"Or use the automatic download script:\n\n"
                f"  Windows PowerShell:\n"
                f"    cd {os.path.dirname(checkpoint_path)}\n"
                f"    Invoke-WebRequest -Uri 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip' -OutFile 'models.zip'\n"
                f"    Expand-Archive -Path 'models.zip' -DestinationPath '.' -Force\n\n"
                f"  Linux/Mac:\n"
                f"    cd {os.path.dirname(checkpoint_path)}\n"
                f"    wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip\n"
                f"    unzip models.zip\n"
            )

        print(f"[RAFT Loader] Loading model: {model_name} from {checkpoint_path}")

        # Create model with appropriate architecture
        args = argparse.Namespace()
        args.small = (model_name == "raft-small")
        args.mixed_precision = False
        args.alternate_corr = False

        model = RAFT_Original(args)

        # Load weights from checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Strip any 'module.' prefixes from DataParallel checkpoints
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = OrderedDict(
                    (key.replace("module.", "", 1), value) for key, value in state_dict.items()
                )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint incompatible with RAFT architecture.\n"
                    f"Missing keys: {sorted(missing)}\n"
                    f"Unexpected keys: {sorted(unexpected)}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RAFT checkpoint: {checkpoint_path}\n"
                f"Error: {e}\n"
                f"The checkpoint file may be corrupted. Try re-downloading."
            )

        # Move to device and set to eval mode
        model = model.to(device).eval()

        # Cache the model
        cls._cache[cache_key] = model
        print(f"[RAFT Loader] Successfully loaded {model_name} on {device}")

        return model

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._cache.clear()
        print("[RAFT Loader] Cache cleared")


# Export public API
__all__ = ['RAFTLoader', 'RAFT_Original']

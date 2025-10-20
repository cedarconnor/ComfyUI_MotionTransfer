"""ComfyUI Motion Transfer Node Pack

Transfer motion from low-resolution AI-generated videos to ultra-high-resolution still images.

Supports three pipelines:
- Pipeline A (Flow-Warp): Dense optical flow -> STMap -> tiled warping
- Pipeline B (Mesh-Warp): Flow -> mesh deformation -> barycentric warping
- Pipeline C (3D-Proxy): Depth estimation -> 3D proxy reprojection

Author: AI-assisted development
License: MIT
"""

from .motion_transfer_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "0.1.0"

# Expose for ComfyUI auto-discovery
WEB_DIRECTORY = "./web"  # For future web UI components

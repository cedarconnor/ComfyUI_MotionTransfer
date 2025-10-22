"""ComfyUI Motion Transfer Node Pack

Transfer motion from low-resolution AI-generated videos to ultra-high-resolution still images.

Supports three pipelines:
- Pipeline A (Flow-Warp): Dense optical flow -> STMap -> tiled warping
- Pipeline B (Mesh-Warp): Flow -> mesh deformation -> barycentric warping
- Pipeline B2 (CoTracker Mesh-Warp): Point tracking -> mesh -> barycentric warping
- Pipeline C (3D-Proxy): Depth estimation -> 3D proxy reprojection (experimental)

Features (v0.6.0):
- Bundled RAFT and SEA-RAFT optical flow models
- Modular model loading architecture
- HuggingFace Hub auto-download for SEA-RAFT
- 2.3x faster performance with SEA-RAFT

Author: AI-assisted development
License: MIT (RAFT/SEA-RAFT vendor code: BSD-3-Clause)
"""

from .motion_transfer_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "0.6.0"

# Expose for ComfyUI auto-discovery
WEB_DIRECTORY = "./web"  # For future web UI components

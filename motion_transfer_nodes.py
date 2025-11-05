"""
ComfyUI Motion Transfer Node Pack - Compatibility Shim

This file maintains backward compatibility with existing imports.
All actual node implementations have been moved to the nodes/ package for better organization.

New modular structure:
- nodes/flow_nodes.py - Optical flow extraction and processing
- nodes/warp_nodes.py - Image warping and output
- nodes/mesh_nodes.py - Mesh generation and barycentric warping
- nodes/depth_nodes.py - Depth estimation and 3D reprojection
- nodes/sequential_node.py - Combined sequential processing

For new code, prefer importing from nodes package directly:
    from nodes.flow_nodes import RAFTFlowExtractor
    from nodes import NODE_CLASS_MAPPINGS
"""

from PIL import Image

# Disable PIL decompression bomb protection for ultra-high-resolution images
# This package is designed for 16K+ images (up to ~300 megapixels)
Image.MAX_IMAGE_PIXELS = None

# Re-export everything from nodes package for backward compatibility
from .nodes import (
    # Node classes
    RAFTFlowExtractor,
    FlowSRRefine,
    FlowToSTMap,
    TileWarp16K,
    TemporalConsistency,
    HiResWriter,
    MeshBuilder2D,
    AdaptiveTessellate,
    MeshFromCoTracker,
    BarycentricWarp,
    DepthEstimator,
    ProxyReprojector,
    SequentialMotionTransfer,

    # ComfyUI registration mappings
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Export all for backward compatibility
__all__ = [
    "RAFTFlowExtractor",
    "FlowSRRefine",
    "FlowToSTMap",
    "TileWarp16K",
    "TemporalConsistency",
    "HiResWriter",
    "MeshBuilder2D",
    "AdaptiveTessellate",
    "MeshFromCoTracker",
    "BarycentricWarp",
    "DepthEstimator",
    "ProxyReprojector",
    "SequentialMotionTransfer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# Log info message on first import
import sys
from .utils.logger import get_logger

if __name__ != "__main__":
    _module_name = __name__
    if not getattr(sys.modules.get(_module_name), '_import_message_shown', False):
        logger = get_logger()
        logger.debug("Using modular nodes/ package (v0.8+)")
        setattr(sys.modules[_module_name], '_import_message_shown', True)

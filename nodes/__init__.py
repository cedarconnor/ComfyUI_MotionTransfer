"""
ComfyUI Motion Transfer Nodes Package

Modular organization of motion transfer nodes into logical groups:
- flow_nodes: Optical flow extraction and processing
- warp_nodes: Image warping and output
- mesh_nodes: Mesh generation and barycentric warping
- depth_nodes: Depth estimation and 3D reprojection
- sequential_node: Combined sequential processing
"""

# Import all node classes
from .flow_nodes import RAFTFlowExtractor, FlowSRRefine, FlowToSTMap
from .warp_nodes import TileWarp16K, TemporalConsistency, HiResWriter
from .mesh_nodes import MeshBuilder2D, AdaptiveTessellate, MeshFromCoTracker, BarycentricWarp
from .depth_nodes import DepthEstimator, ProxyReprojector
from .sequential_node import SequentialMotionTransfer

# Build node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    # Flow nodes (Pipeline A)
    "RAFTFlowExtractor": RAFTFlowExtractor,
    "FlowSRRefine": FlowSRRefine,
    "FlowToSTMap": FlowToSTMap,

    # Warp nodes (Pipeline A)
    "TileWarp16K": TileWarp16K,
    "TemporalConsistency": TemporalConsistency,
    "HiResWriter": HiResWriter,

    # Mesh nodes (Pipeline B & B2)
    "MeshBuilder2D": MeshBuilder2D,
    "AdaptiveTessellate": AdaptiveTessellate,
    "MeshFromCoTracker": MeshFromCoTracker,
    "BarycentricWarp": BarycentricWarp,

    # Depth nodes (Pipeline C)
    "DepthEstimator": DepthEstimator,
    "ProxyReprojector": ProxyReprojector,

    # Combined node
    "SequentialMotionTransfer": SequentialMotionTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Flow nodes
    "RAFTFlowExtractor": "RAFT Flow Extractor",
    "FlowSRRefine": "Flow SR Refine",
    "FlowToSTMap": "Flow to STMap",

    # Warp nodes
    "TileWarp16K": "Tile Warp 16K",
    "TemporalConsistency": "Temporal Consistency",
    "HiResWriter": "Hi-Res Writer",

    # Mesh nodes
    "MeshBuilder2D": "Mesh Builder 2D",
    "AdaptiveTessellate": "Adaptive Tessellate",
    "MeshFromCoTracker": "Mesh from CoTracker",
    "BarycentricWarp": "Barycentric Warp",

    # Depth nodes
    "DepthEstimator": "Depth Estimator",
    "ProxyReprojector": "Proxy Reprojector",

    # Combined node
    "SequentialMotionTransfer": "Sequential Motion Transfer",
}

# Export all
__all__ = [
    # Classes
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

    # Mappings
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

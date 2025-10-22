"""CUDA acceleration package for Motion Transfer nodes.

This package provides GPU-accelerated implementations of critical nodes:
- TileWarp16K: 8-15× speedup
- BarycentricWarp: 10-20× speedup
- FlowSRRefine: 3-5× speedup

If CUDA is unavailable, nodes gracefully fall back to CPU implementation.
"""

from . import cuda_loader

__all__ = ['cuda_loader']

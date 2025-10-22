"""
CUDA Kernel Loader for Motion Transfer Nodes

Dynamically loads compiled CUDA kernels (.so/.dll) and provides Python interface.
Gracefully falls back to CPU if CUDA is unavailable.

Author: AI-assisted implementation
License: MIT
"""

import os
import sys
import ctypes
import numpy as np
import torch

# Global state
_cuda_available = False
_lib_tile_warp = None
_lib_barycentric = None
_lib_guided_filter = None


def _find_cuda_lib(lib_name):
    """Find compiled CUDA library (.so on Linux, .dll on Windows)."""
    cuda_dir = os.path.dirname(__file__)

    if sys.platform == "win32":
        lib_file = f"{lib_name}.dll"
    else:
        lib_file = f"lib{lib_name}.so"

    lib_path = os.path.join(cuda_dir, lib_file)

    if os.path.exists(lib_path):
        return lib_path
    return None


def init_cuda_kernels():
    """Initialize CUDA kernels (load compiled libraries)."""
    global _cuda_available, _lib_tile_warp, _lib_barycentric, _lib_guided_filter

    if not torch.cuda.is_available():
        print("[Motion Transfer CUDA] CUDA not available, using CPU fallback")
        return False

    try:
        # Load TileWarp kernel
        tile_warp_path = _find_cuda_lib("tile_warp_kernel")
        if tile_warp_path:
            _lib_tile_warp = ctypes.CDLL(tile_warp_path)

            # Define function signatures
            _lib_tile_warp.bind_source_texture.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            _lib_tile_warp.warp_tile_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            _lib_tile_warp.normalize_output_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            print("[Motion Transfer CUDA] Loaded TileWarp kernel")

        # Load BarycentricWarp kernel
        barycentric_path = _find_cuda_lib("barycentric_warp_kernel")
        if barycentric_path:
            _lib_barycentric = ctypes.CDLL(barycentric_path)

            _lib_barycentric.bind_mesh_source_texture.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            _lib_barycentric.rasterize_mesh_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            _lib_barycentric.normalize_mesh_output_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            print("[Motion Transfer CUDA] Loaded BarycentricWarp kernel")

        # Load GuidedFilter kernel
        guided_filter_path = _find_cuda_lib("guided_filter_kernel")
        if guided_filter_path:
            _lib_guided_filter = ctypes.CDLL(guided_filter_path)

            _lib_guided_filter.guided_filter_cuda.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_float
            ]
            print("[Motion Transfer CUDA] Loaded GuidedFilter kernel")

        _cuda_available = (_lib_tile_warp is not None or
                          _lib_barycentric is not None or
                          _lib_guided_filter is not None)

        if _cuda_available:
            print("[Motion Transfer CUDA] Initialized successfully")
            return True
        else:
            print("[Motion Transfer CUDA] No CUDA kernels found, using CPU fallback")
            return False

    except Exception as e:
        print(f"[Motion Transfer CUDA] Failed to load kernels: {e}")
        print("[Motion Transfer CUDA] Using CPU fallback")
        return False


def is_cuda_available():
    """Check if CUDA kernels are available."""
    return _cuda_available


# ------------------------------------------------------------
# TileWarp CUDA API
# ------------------------------------------------------------

class CUDATileWarp:
    """CUDA-accelerated tile warping with STMaps."""

    def __init__(self, still_image_np, use_bicubic=False):
        """
        Initialize with source image.

        Args:
            still_image_np: numpy array [H, W, C] (float32)
            use_bicubic: If True, use bicubic interpolation (slower but higher quality)
        """
        if _lib_tile_warp is None:
            raise RuntimeError("TileWarp CUDA kernel not loaded")

        self.h, self.w, self.c = still_image_np.shape
        self.use_bicubic = use_bicubic

        # Bind source image to CUDA texture
        img_ptr = still_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib_tile_warp.bind_source_texture(img_ptr, self.w, self.h, self.c)

        # Allocate device memory for output and weights (reused across tiles)
        self.d_output = torch.zeros((self.h, self.w, 4), dtype=torch.float32, device='cuda')
        self.d_weights = torch.zeros((self.h, self.w), dtype=torch.float32, device='cuda')

    def warp_tile(self, stmap_tile, feather_mask, tile_x0, tile_y0):
        """
        Warp a single tile with feather blending.

        Args:
            stmap_tile: numpy array [tile_h, tile_w, 3] (STMap coordinates)
            feather_mask: numpy array [tile_h, tile_w] (feather weights)
            tile_x0, tile_y0: Tile origin in output image
        """
        tile_h, tile_w = stmap_tile.shape[:2]

        stmap_ptr = stmap_tile.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        feather_ptr = feather_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_ptr = self.d_output.data_ptr()
        weight_ptr = self.d_weights.data_ptr()

        _lib_tile_warp.warp_tile_cuda(
            stmap_ptr, feather_ptr,
            ctypes.c_void_p(out_ptr), ctypes.c_void_p(weight_ptr),
            tile_x0, tile_y0, tile_w, tile_h,
            self.w, self.h, int(self.use_bicubic)
        )

    def finalize(self):
        """
        Normalize output by accumulated weights.

        Returns:
            numpy array [H, W, C] (final warped image)
        """
        out_ptr = self.d_output.data_ptr()
        weight_ptr = self.d_weights.data_ptr()

        _lib_tile_warp.normalize_output_cuda(
            ctypes.c_void_p(out_ptr), ctypes.c_void_p(weight_ptr),
            self.w, self.h
        )

        # Convert back to numpy (trim alpha channel if needed)
        result = self.d_output.cpu().numpy()
        if self.c == 3:
            result = result[:, :, :3]
        return result


# ------------------------------------------------------------
# BarycentricWarp CUDA API
# ------------------------------------------------------------

class CUDABarycentricWarp:
    """CUDA-accelerated barycentric mesh warping."""

    def __init__(self, still_image_np):
        """
        Initialize with source image.

        Args:
            still_image_np: numpy array [H, W, C] (float32)
        """
        if _lib_barycentric is None:
            raise RuntimeError("BarycentricWarp CUDA kernel not loaded")

        self.h, self.w, self.c = still_image_np.shape

        # Bind source image to texture
        img_ptr = still_image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib_barycentric.bind_mesh_source_texture(img_ptr, self.w, self.h, self.c)

        # Allocate output buffers
        self.d_output = torch.zeros((self.h, self.w, 4), dtype=torch.float32, device='cuda')
        self.d_coverage = torch.zeros((self.h, self.w), dtype=torch.float32, device='cuda')

    def warp_mesh(self, dst_vertices, src_vertices, num_triangles):
        """
        Rasterize mesh triangles.

        Args:
            dst_vertices: numpy array [num_tri, 3, 2] (deformed positions)
            src_vertices: numpy array [num_tri, 3, 2] (source UV coordinates)
            num_triangles: Number of triangles

        Returns:
            numpy array [H, W, C] (warped image)
        """
        # Flatten vertices to [num_tri * 6] for C API
        dst_flat = dst_vertices.reshape(-1).astype(np.float32)
        src_flat = src_vertices.reshape(-1).astype(np.float32)

        dst_ptr = dst_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        src_ptr = src_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_ptr = self.d_output.data_ptr()
        cov_ptr = self.d_coverage.data_ptr()

        _lib_barycentric.rasterize_mesh_cuda(
            dst_ptr, src_ptr,
            ctypes.c_void_p(out_ptr), ctypes.c_void_p(cov_ptr),
            self.w, self.h, num_triangles
        )

        # Normalize by coverage
        _lib_barycentric.normalize_mesh_output_cuda(
            ctypes.c_void_p(out_ptr), ctypes.c_void_p(cov_ptr),
            self.w, self.h
        )

        # Convert to numpy
        result = self.d_output.cpu().numpy()
        if self.c == 3:
            result = result[:, :, :3]
        return result


# ------------------------------------------------------------
# GuidedFilter CUDA API
# ------------------------------------------------------------

def guided_filter_cuda(guide_image_np, flow_np, radius, eps):
    """
    Apply CUDA-accelerated guided filter.

    Args:
        guide_image_np: numpy array [H, W] (grayscale guide, float32)
        flow_np: numpy array [H, W] (flow channel to filter, float32)
        radius: Filter radius (int)
        eps: Regularization parameter (float)

    Returns:
        numpy array [H, W] (filtered flow)
    """
    if _lib_guided_filter is None:
        raise RuntimeError("GuidedFilter CUDA kernel not loaded")

    h, w = guide_image_np.shape

    # Upload to GPU
    d_guide = torch.from_numpy(guide_image_np).cuda()
    d_flow = torch.from_numpy(flow_np).cuda()
    d_output = torch.zeros_like(d_flow)

    guide_ptr = d_guide.data_ptr()
    flow_ptr = d_flow.data_ptr()
    out_ptr = d_output.data_ptr()

    _lib_guided_filter.guided_filter_cuda(
        ctypes.c_void_p(guide_ptr), ctypes.c_void_p(flow_ptr),
        ctypes.c_void_p(out_ptr), w, h, radius, eps
    )

    return d_output.cpu().numpy()


# Auto-initialize on import
init_cuda_kernels()

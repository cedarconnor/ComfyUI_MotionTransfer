"""
Warping and output nodes.

Contains nodes for tiled warping, temporal consistency, and high-resolution output writing.
"""

import torch
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, List, Optional

# Try to import CUDA accelerated kernels
try:
    from ..cuda import cuda_loader
    CUDA_AVAILABLE = cuda_loader.is_cuda_available()
except ImportError:
    CUDA_AVAILABLE = False

# Import logger
from ..utils.logger import get_logger

logger = get_logger()


class TileWarp16K:
    """Apply STMap warping to ultra-high-resolution images using tiled processing with feathered blending.

    Handles 16K+ images by processing in tiles with overlap, using linear feathering
    to ensure seamless stitching across tile boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "still_image": ("IMAGE", {
                    "tooltip": "High-resolution still image to warp. This is the 16K (or other high-res) image that will have motion applied to it."
                }),
                "stmap": ("IMAGE", {
                    "tooltip": "STMap sequence from FlowToSTMap. Contains normalized UV coordinates that define how to warp each pixel."
                }),
                "tile_size": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "tooltip": "Size of processing tiles. Larger tiles (4096) are faster but need more VRAM. Use 2048 for 24GB GPU, 1024 for 12GB GPU, 512 for 8GB GPU."
                }),
                "overlap": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 512,
                    "tooltip": "Overlap between tiles for blending. Larger values (256) give smoother seams but slower processing. 128 is recommended, use 64 minimum."
                }),
                "interpolation": (["cubic", "linear", "lanczos4"], {
                    "default": "cubic",
                    "tooltip": "Interpolation method. 'cubic': best quality/speed balance (recommended). 'linear': fastest but lower quality. 'lanczos4': highest quality but slowest."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_sequence",)
    FUNCTION = "warp"
    CATEGORY = "MotionTransfer/Warp"

    def warp(self, still_image, stmap, tile_size, overlap, interpolation):
        """Apply STMap warping with tiled processing and feathered blending.

        Args:
            still_image: [1, H, W, C] high-resolution still
            stmap: [B, H, W, 3] STMap sequence
            tile_size: Size of processing tiles
            overlap: Overlap between tiles for blending
            interpolation: Interpolation method

        Returns:
            warped_sequence: [B, H, W, C] warped frames
        """
        # Validate overlap < tile_size to prevent infinite loop
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than tile_size ({tile_size}). "
                f"This would cause step = tile_size - overlap = {tile_size - overlap}, "
                f"freezing the tiling loop. Please reduce overlap or increase tile_size."
            )

        if isinstance(still_image, torch.Tensor):
            still_image = still_image.cpu().numpy()
        if isinstance(stmap, torch.Tensor):
            stmap = stmap.cpu().numpy()

        # Get still image (first frame if batch)
        still = still_image[0] if len(still_image.shape) == 4 else still_image
        h, w, c = still.shape

        # Try CUDA acceleration first
        if CUDA_AVAILABLE and torch.cuda.is_available():
            try:
                return self._warp_cuda(still, stmap, tile_size, overlap, interpolation)
            except Exception as e:
                print(f"[TileWarp16K] CUDA failed ({e}), falling back to CPU")

        # CPU fallback
        return self._warp_cpu(still, stmap, tile_size, overlap, interpolation)

    def _warp_cuda(self, still, stmap, tile_size, overlap, interpolation):
        """CUDA-accelerated warping (8-15× faster than CPU)."""
        h, w, c = still.shape
        batch_size = stmap.shape[0]
        use_bicubic = (interpolation == "cubic" or interpolation == "lanczos4")

        # Initialize CUDA warper
        warp_engine = cuda_loader.CUDATileWarp(still.astype(np.float32), use_bicubic=use_bicubic)

        warped_frames = []
        for frame_idx in range(batch_size):
            stmap_frame = stmap[frame_idx]  # [H, W, 3]

            # Process tiles
            step = tile_size - overlap
            for y0 in range(0, h, step):
                for x0 in range(0, w, step):
                    y1 = min(y0 + tile_size, h)
                    x1 = min(x0 + tile_size, w)
                    tile_h = y1 - y0
                    tile_w = x1 - x0

                    stmap_tile = stmap_frame[y0:y1, x0:x1]

                    # Get feather mask
                    tile_feather = self._get_tile_feather(
                        tile_h, tile_w, tile_size, overlap,
                        is_top=(y0 == 0), is_left=(x0 == 0),
                        is_bottom=(y1 == h), is_right=(x1 == w)
                    )

                    # CUDA tile warp
                    warp_engine.warp_tile(
                        stmap_tile.astype(np.float32),
                        tile_feather.astype(np.float32),
                        x0, y0
                    )

            # Finalize (normalize by weights)
            warped_full = warp_engine.finalize()
            warped_frames.append(warped_full[:, :, :c])  # Trim to original channels

        result = np.stack(warped_frames, axis=0)
        return (result,)

    def _warp_cpu(self, still, stmap, tile_size, overlap, interpolation):
        """CPU fallback (original implementation)."""
        h, w, c = still.shape

        # Get interpolation mode
        interp_map = {
            "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        interp_mode = interp_map[interpolation]

        # Process each STMap frame
        batch_size = stmap.shape[0]
        warped_frames = []

        for frame_idx in range(batch_size):
            stmap_frame = stmap[frame_idx]  # [H, W, 3]

            # Initialize output and weight accumulation buffers
            warped_full = np.zeros((h, w, c), dtype=np.float32)
            weight_full = np.zeros((h, w, 1), dtype=np.float32)

            # Tile processing
            step = tile_size - overlap
            for y0 in range(0, h, step):
                for x0 in range(0, w, step):
                    # Tile boundaries
                    y1 = min(y0 + tile_size, h)
                    x1 = min(x0 + tile_size, w)
                    tile_h = y1 - y0
                    tile_w = x1 - x0

                    # Extract tiles
                    stmap_tile = stmap_frame[y0:y1, x0:x1]

                    # Create remap coordinates (denormalize STMap)
                    map_x = (stmap_tile[:, :, 0] * (w - 1)).astype(np.float32)
                    map_y = (stmap_tile[:, :, 1] * (h - 1)).astype(np.float32)

                    # Apply warp to tile
                    warped_tile = cv2.remap(
                        still,  # Use full image for source to handle flow outside tile
                        map_x, map_y,
                        interpolation=interp_mode,
                        borderMode=cv2.BORDER_REFLECT_101
                    )

                    # Get feather mask for this tile
                    tile_feather = self._get_tile_feather(
                        tile_h, tile_w, tile_size, overlap,
                        is_top=(y0 == 0), is_left=(x0 == 0),
                        is_bottom=(y1 == h), is_right=(x1 == w)
                    )

                    # Accumulate with feathered blending
                    warped_full[y0:y1, x0:x1] += warped_tile * tile_feather
                    weight_full[y0:y1, x0:x1] += tile_feather

            # Normalize by weights
            warped_full = np.divide(
                warped_full, weight_full,
                out=np.zeros_like(warped_full),
                where=weight_full > 0
            )

            warped_frames.append(warped_full.astype(np.float32))

        result = np.stack(warped_frames, axis=0)  # [B, H, W, C]
        return (result,)

    def _create_feather_mask(self, tile_size, overlap):
        """Create feather weight mask for tile blending."""
        # Not used directly, but kept for reference
        return None

    def _get_tile_feather(self, tile_h, tile_w, tile_size, overlap, is_top, is_left, is_bottom, is_right):
        """Generate feather mask for a specific tile position.

        Args:
            tile_h, tile_w: Actual tile dimensions
            tile_size: Nominal tile size
            overlap: Overlap width
            is_top, is_left, is_bottom, is_right: Edge flags

        Returns:
            feather: [H, W, 1] weight mask with linear gradients in overlap regions
        """
        feather = np.ones((tile_h, tile_w, 1), dtype=np.float32)

        # Create linear ramps for each edge with correct broadcasting
        if not is_left and overlap > 0:
            # Left edge fade-in: shape (tile_w,) broadcast to (tile_h, tile_w, 1)
            ramp_len = min(overlap, tile_w)
            ramp = np.linspace(0, 1, ramp_len)
            # Reshape to (1, ramp_len, 1) for proper broadcasting
            feather[:, :ramp_len, :] *= ramp[None, :, None]

        if not is_right and overlap > 0:
            # Right edge fade-out
            ramp_len = min(overlap, tile_w)
            ramp = np.linspace(1, 0, ramp_len)
            # Reshape to (1, ramp_len, 1)
            feather[:, -ramp_len:, :] *= ramp[None, :, None]

        if not is_top and overlap > 0:
            # Top edge fade-in: shape (tile_h,) broadcast to (tile_h, tile_w, 1)
            ramp_len = min(overlap, tile_h)
            ramp = np.linspace(0, 1, ramp_len)
            # Reshape to (ramp_len, 1, 1)
            feather[:ramp_len, :, :] *= ramp[:, None, None]

        if not is_bottom and overlap > 0:
            # Bottom edge fade-out
            ramp_len = min(overlap, tile_h)
            ramp = np.linspace(1, 0, ramp_len)
            # Reshape to (ramp_len, 1, 1)
            feather[-ramp_len:, :, :] *= ramp[:, None, None]

        return feather


# ------------------------------------------------------
# Node 6: TemporalConsistency - Temporal stabilization
# ------------------------------------------------------
class TemporalConsistency:
    """Apply temporal stabilization using flow-based frame blending.

    Reduces flicker and jitter by blending each frame with the previous frame
    warped forward using optical flow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Warped frame sequence from TileWarp16K. These frames will be temporally stabilized to reduce flicker."
                }),
                "flow": ("FLOW", {
                    "tooltip": "High-resolution flow fields from FlowSRRefine. Used to warp previous frame forward for temporal blending."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Temporal blending strength. 0.0 = no blending (may flicker), 0.3 = balanced (recommended), 0.5+ = strong smoothing (may blur motion). Reduce if motion looks ghosted."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stabilized",)
    FUNCTION = "stabilize"
    CATEGORY = "MotionTransfer/Temporal"

    def stabilize(self, frames, flow, blend_strength):
        """Apply temporal blending for flicker reduction.

        Args:
            frames: [B, H, W, C] frame sequence
            flow: [B-1, H, W, 2] forward flow fields between consecutive frames
            blend_strength: Blending weight for previous frame [0=none, 1=full]

        Returns:
            stabilized: [B, H, W, C] temporally stabilized frames
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        batch_size = frames.shape[0]
        flow_count = flow.shape[0]
        h, w = frames.shape[1:3]

        # Validate flow array size
        if flow_count != batch_size - 1:
            raise ValueError(
                f"Flow array size mismatch: expected {batch_size - 1} flow fields "
                f"for {batch_size} frames, but got {flow_count}. "
                f"Flow should contain transitions between consecutive frames (B-1 entries for B frames)."
            )

        stabilized = [frames[0]]  # First frame unchanged

        for t in range(1, batch_size):
            current_frame = frames[t]
            prev_stabilized = stabilized[-1]
            # Flow index: flow[0] is frame0→frame1, flow[1] is frame1→frame2, etc.
            # For frame t, we need flow from frame(t-1) to frame(t), which is flow[t-1]
            flow_fwd = flow[t-1]  # Forward flow from t-1 to t

            # To warp previous frame forward, we need inverse mapping
            # Forward flow tells us where pixels move TO, but remap needs where to sample FROM
            # So we use the inverse: sample from (position - flow)
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x - flow_fwd[:, :, 0]).astype(np.float32)  # Inverse warp
            map_y = (map_y - flow_fwd[:, :, 1]).astype(np.float32)  # Inverse warp

            warped_prev = cv2.remap(
                prev_stabilized, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Blend current with warped previous
            blended = cv2.addWeighted(
                current_frame.astype(np.float32), 1.0 - blend_strength,
                warped_prev.astype(np.float32), blend_strength,
                0
            )

            stabilized.append(blended.astype(np.float32))

        result = np.stack(stabilized, axis=0)
        return (result,)


# ------------------------------------------------------
# Node 7: HiResWriter - Export high-res sequences
# ------------------------------------------------------
class HiResWriter:
    """Write high-resolution image sequences to disk as individual frames.

    Supports PNG, EXR, and JPG formats. For video encoding, use external tools
    like FFmpeg on the exported frame sequence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Final image sequence to export (typically from TemporalConsistency). Will be written to disk as individual frames."
                }),
                "output_path": ("STRING", {
                    "default": "output/frame",
                    "tooltip": "Output file path pattern (without extension). Example: 'C:/renders/shot01/frame' will create frame_0000.png, frame_0001.png, etc. Directory will be created if needed."
                }),
                "format": (["png", "exr", "jpg"], {
                    "default": "png",
                    "tooltip": "Output format. 'png': 8-bit sRGB, lossless (recommended for web/preview). 'exr': 16-bit half float, linear (best for VFX/compositing). 'jpg': 8-bit sRGB, quality 95 (smallest files)."
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame number for file naming. Use 0 for frame_0000, 1001 for film standard (frame_1001), etc."
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_sequence"
    CATEGORY = "MotionTransfer/IO"

    def write_sequence(self, images, output_path, format, start_frame):
        """Write image sequence to disk.

        Args:
            images: [B, H, W, C] image sequence
            output_path: Output path pattern (e.g., "output/frame")
            format: Output format (png, exr, jpg)
            start_frame: Starting frame number for naming

        Returns:
            Empty tuple (output node)
        """
        import os
        from pathlib import Path

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = images.shape[0]

        for i in range(batch_size):
            frame_num = start_frame + i
            frame = images[i]

            # Build filename
            if format == "exr":
                filename = f"{output_path}_{frame_num:04d}.exr"
                self._write_exr(frame, filename)
            elif format == "png":
                filename = f"{output_path}_{frame_num:04d}.png"
                self._write_png(frame, filename)
            elif format == "jpg":
                filename = f"{output_path}_{frame_num:04d}.jpg"
                self._write_jpg(frame, filename)

            print(f"Wrote frame {frame_num}: {filename}")

        print(f"Wrote {batch_size} frames to {output_dir}")
        return ()

    def _write_exr(self, image, filename):
        """Write EXR file (float16 half precision)."""
        try:
            import OpenEXR
            import Imath

            h, w, c = image.shape
            header = OpenEXR.Header(w, h)
            half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
            header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}

            # Convert to float16 for half precision (must match HALF channel type)
            image_f16 = image.astype(np.float16)
            r = image_f16[:, :, 0].flatten().tobytes()
            g = image_f16[:, :, 1].flatten().tobytes()
            b = image_f16[:, :, 2].flatten().tobytes()

            exr = OpenEXR.OutputFile(filename, header)
            exr.writePixels({'R': r, 'G': g, 'B': b})
            exr.close()
        except ImportError:
            print("WARNING: OpenEXR not installed, falling back to PNG")
            self._write_png(image, filename.replace('.exr', '.png'))

    def _write_png(self, image, filename):
        """Write PNG file (8-bit)."""
        img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_bgr)

    def _write_jpg(self, image, filename):
        """Write JPG file (8-bit)."""
        img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


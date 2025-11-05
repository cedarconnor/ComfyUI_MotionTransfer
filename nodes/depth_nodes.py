"""
Depth estimation and 3D reprojection nodes.

Contains nodes for monocular depth estimation and depth-based proxy reprojection
for parallax handling (Pipeline C - experimental).
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional


class DepthEstimator:
    """Estimate depth maps from video frames using monocular depth estimation.

    Useful for handling parallax and camera motion in the source video.
    """

    _model = None
    _model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames from ComfyUI video loader. Depth will be estimated for each frame to enable parallax-aware warping."
                }),
                "model": (["midas", "dpt"], {
                    "default": "midas",
                    "tooltip": "Depth estimation model. 'midas': MiDaS (lighter, faster). 'dpt': DPT (more accurate). Currently placeholder - real models not yet integrated, uses simple Gaussian blur."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "estimate_depth"
    CATEGORY = "MotionTransfer/Depth"

    def estimate_depth(self, images, model):
        """Estimate depth maps from images.

        Args:
            images: [B, H, W, C] input frames
            model: Depth estimation model to use

        Returns:
            depth_maps: [B, H, W, 1] normalized depth maps
        """
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = images

        # Load model (cached)
        depth_model = self._load_model(model)

        depth_maps = []
        for i in range(images_np.shape[0]):
            frame = images_np[i]

            # Simple placeholder depth estimation
            # In production, would use MiDaS or DPT models
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            depth = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32) / 255.0

            depth_maps.append(depth[:, :, None])

        result = np.stack(depth_maps, axis=0)
        return (result,)

    @classmethod
    def _load_model(cls, model_name):
        """Load depth estimation model."""
        if cls._model is None or cls._model_name != model_name:
            # Placeholder - in production would load actual MiDaS/DPT model
            print(f"Loading depth model: {model_name}")
            cls._model = "placeholder"
            cls._model_name = model_name
        return cls._model


# ------------------------------------------------------
# Node 12: ProxyReprojector - 3D proxy reprojection
# ------------------------------------------------------
class ProxyReprojector:
    """Reproject texture onto 3D proxy geometry using depth and camera motion.

    Handles parallax by treating the scene as a depth-based 3D proxy.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "still_image": ("IMAGE", {
                    "tooltip": "High-resolution still image to reproject using depth information. Depth enables parallax-correct warping for camera motion."
                }),
                "depth_maps": ("IMAGE", {
                    "tooltip": "Depth map sequence from DepthEstimator. Closer objects (brighter) move more than distant objects (darker) under camera motion."
                }),
                "flow": ("FLOW", {
                    "tooltip": "Optical flow from RAFTFlowExtractor. Combined with depth to estimate camera motion and create parallax-aware warping."
                }),
                "focal_length": ("FLOAT", {
                    "default": 1000.0,
                    "min": 100.0,
                    "max": 10000.0,
                    "tooltip": "Estimated camera focal length in pixels. Higher values (2000-5000) = telephoto (less perspective). Lower values (500-1000) = wide angle (more perspective). Affects parallax strength."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reprojected_sequence",)
    FUNCTION = "reproject"
    CATEGORY = "MotionTransfer/Depth"

    def reproject(self, still_image, depth_maps, flow, focal_length):
        """Reproject still image using depth proxy.

        Args:
            still_image: [1, H, W, C] source texture
            depth_maps: [B, H, W, 1] depth sequence (should match still resolution)
            flow: [B-1 or B, H, W, 2] optical flow - MUST be upscaled to still resolution first!
            focal_length: Camera focal length estimate

        Returns:
            reprojected_sequence: [B, H, W, C] reprojected frames

        IMPORTANT: Flow must be pre-upscaled to match still_image resolution!
        Use FlowSRRefine before this node to upscale flow to high-res.
        """
        if isinstance(still_image, torch.Tensor):
            still_image = still_image.cpu().numpy()
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        still = still_image[0] if len(still_image.shape) == 4 else still_image
        h, w = still.shape[:2]

        reprojected = []
        num_frames = depth_maps.shape[0]

        for i in range(num_frames):
            depth = depth_maps[i, :, :, 0]

            # RAFT produces (B-1) flows for B images: flow[j] goes from frame j to j+1
            # Frame 0: no incoming flow, use identity
            # Frame i (i>0): use flow[i-1] which represents motion from frame i-1 to i
            if i == 0:
                flow_frame = np.zeros((h, w, 2), dtype=np.float32)
            else:
                flow_frame = flow[i - 1]

                # Verify flow resolution matches still resolution
                if flow_frame.shape[0] != h or flow_frame.shape[1] != w:
                    raise ValueError(
                        f"Flow resolution mismatch! Flow is {flow_frame.shape[0]}x{flow_frame.shape[1]} "
                        f"but still image is {h}x{w}. You must upscale flow first using FlowSRRefine node.\n\n"
                        f"Correct pipeline: RAFTFlowExtractor → FlowSRRefine → ProxyReprojector\n"
                        f"Current (wrong): RAFTFlowExtractor → ProxyReprojector"
                    )

            # Simple parallax-based warping (placeholder for full 3D reprojection)
            # In production, would solve camera pose and do proper 3D reprojection
            scale_map = 1.0 + (depth - 0.5) * 0.2  # Depth-based scaling

            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow_frame[:, :, 0] * scale_map).astype(np.float32)
            map_y = (map_y + flow_frame[:, :, 1] * scale_map).astype(np.float32)

            warped = cv2.remap(
                still, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            reprojected.append(warped)

        result = np.stack(reprojected, axis=0)
        return (result,)


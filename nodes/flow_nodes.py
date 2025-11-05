"""
Optical flow extraction and processing nodes.

Contains nodes for RAFT/SEA-RAFT flow extraction, flow upsampling with guided filtering,
and flow-to-STMap conversion for warping.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional

# Import unified model loader
from ..models import OpticalFlowModel

# Import logger
from ..utils.logger import get_logger

logger = get_logger()


class RAFTFlowExtractor:
    """Extract dense optical flow between consecutive frames using RAFT or SEA-RAFT.

    Supports both original RAFT (2020) and SEA-RAFT (2024 ECCV - 2.3x faster, 22% more accurate).
    Returns flow fields and confidence/uncertainty maps for motion transfer pipeline.
    """

    _model = None
    _model_path = None
    _model_type = None  # Track whether loaded model is 'raft' or 'searaft'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames from ComfyUI video loader. Expects [B, H, W, C] batch of images."
                }),
                "raft_iters": ("INT", {
                    "default": 12,
                    "min": 6,
                    "max": 32,
                    "tooltip": "Refinement iterations. SEA-RAFT needs fewer (6-8) than RAFT (12-20) for same quality. Will auto-adjust to 8 for SEA-RAFT if you leave at default 12."
                }),
                "model_name": ([
                    "raft-sintel",
                    "raft-things",
                    "raft-small",
                    "sea-raft-small",
                    "sea-raft-medium",
                    "sea-raft-large"
                ], {
                    "default": "raft-sintel",
                    "tooltip": "Optical flow model. RAFT: original (2020), requires manual model download. SEA-RAFT: newer (ECCV 2024), 2.3x faster with 22% better accuracy, auto-downloads from HuggingFace. Recommended: sea-raft-medium for best speed/quality balance."
                }),
            }
        }

    RETURN_TYPES = ("FLOW", "IMAGE")  # flow fields [B-1, H, W, 2], confidence [B-1, H, W, 1]
    RETURN_NAMES = ("flow", "confidence")
    FUNCTION = "extract_flow"
    CATEGORY = "MotionTransfer/Flow"

    def extract_flow(self, images, raft_iters, model_name):
        """Extract optical flow between consecutive frame pairs.

        Args:
            images: Tensor [B, H, W, C] in range [0, 1]
            raft_iters: Number of refinement iterations
            model_name: Model variant to use (RAFT or SEA-RAFT)

        Returns:
            flow: Tensor [B-1, H, W, 2] containing (u, v) flow vectors
            confidence: Tensor [B-1, H, W, 1] containing flow confidence/uncertainty scores
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model (RAFT or SEA-RAFT, cached)
        model, model_type = self._load_model(model_name, device)

        # Auto-adjust iterations for SEA-RAFT if user left default
        if model_type == 'searaft' and raft_iters == 12:
            raft_iters = 8
            print(f"[Motion Transfer] Auto-adjusted iterations to {raft_iters} for SEA-RAFT (faster convergence)")

        # Convert ComfyUI format [B, H, W, C] to torch [B, C, H, W]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2).to(device)

        # Extract flow for consecutive pairs
        flows = []
        confidences = []

        with torch.no_grad():
            for i in range(len(images) - 1):
                img1 = images[i:i+1] * 255.0  # RAFT expects [0, 255]
                img2 = images[i+1:i+2] * 255.0

                # Pad to multiple of 8
                from torch.nn.functional import pad
                h, w = img1.shape[2:]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    img1 = pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                    img2 = pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

                # Run model (RAFT or SEA-RAFT)
                if model_type == 'searaft':
                    # SEA-RAFT returns uncertainty as third output
                    flow_low, flow_up, uncertainty = model(img1, img2, iters=raft_iters, test_mode=True)
                else:
                    # Original RAFT returns only flow
                    flow_low, flow_up = model(img1, img2, iters=raft_iters, test_mode=True)
                    uncertainty = None

                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    flow_up = flow_up[:, :, :h, :w]
                    if uncertainty is not None:
                        uncertainty = uncertainty[:, :, :h, :w]

                # Compute confidence
                if model_type == 'searaft' and uncertainty is not None:
                    # Use SEA-RAFT's native uncertainty (better than heuristic)
                    # Uncertainty is already [1, 1, H, W], convert to confidence
                    conf = 1.0 - torch.clamp(uncertainty, 0, 1)
                else:
                    # Use heuristic confidence for original RAFT
                    flow_mag = torch.sqrt(flow_up[:, 0:1]**2 + flow_up[:, 1:2]**2)
                    conf = torch.exp(-flow_mag / 10.0)

                flows.append(flow_up[0].permute(1, 2, 0).cpu())  # [H, W, 2]
                confidences.append(conf[0].permute(1, 2, 0).cpu())  # [H, W, 1]

        # Stack into batch tensors
        flow_batch = torch.stack(flows, dim=0)  # [B-1, H, W, 2]
        conf_batch = torch.stack(confidences, dim=0)  # [B-1, H, W, 1]

        return (flow_batch.numpy(), conf_batch.numpy())

    @classmethod
    def _load_model(cls, model_name, device):
        """Load RAFT or SEA-RAFT model with caching.

        Uses the new unified OpticalFlowModel loader which handles both
        RAFT and SEA-RAFT models cleanly without sys.path manipulation.

        Returns:
            tuple: (model, model_type) where model_type is 'raft' or 'searaft'
        """
        if cls._model is None or cls._model_path != model_name:
            # Use the new unified loader - much simpler!
            model, model_type = OpticalFlowModel.load(model_name, device)

            cls._model = model
            cls._model_path = model_name
            cls._model_type = model_type

        return cls._model, cls._model_type


# ------------------------------------------------------
# Node 3: FlowSRRefine - Upscale and refine flow fields
# ------------------------------------------------------
class FlowSRRefine:
    """Upscale and refine optical flow fields using bicubic interpolation and guided filtering.

    Upscales low-resolution flow to match high-resolution still image, with edge-aware
    smoothing to prevent flow bleeding across sharp boundaries.
    """

    _guided_filter_available = hasattr(cv2, "ximgproc") and hasattr(getattr(cv2, "ximgproc", None), "guidedFilter")
    _guided_filter_warning_shown = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW", {
                    "tooltip": "Optical flow fields from RAFTFlowExtractor. Low-resolution flow to be upscaled."
                }),
                "guide_image": ("IMAGE", {
                    "tooltip": "High-resolution still image used as guidance for edge-aware filtering. Prevents flow from bleeding across sharp edges."
                }),
                "target_width": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target width for upscaled flow (should match your high-res still width). Common: 4K=3840, 8K=7680, 16K=15360."
                }),
                "target_height": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target height for upscaled flow (should match your high-res still height). Common: 4K=2160, 8K=4320, 16K=8640."
                }),
                "guided_filter_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Radius for guided filter smoothing. Larger values (16-32) give smoother flow, smaller values (4-8) preserve detail better. 8 is a good default."
                }),
                "guided_filter_eps": ("FLOAT", {
                    "default": 1e-3,
                    "min": 1e-6,
                    "max": 1.0,
                    "tooltip": "Regularization parameter for guided filter. Lower values (1e-4) preserve edges better, higher values (1e-2) give smoother results. 1e-3 is recommended."
                }),
            }
        }

    RETURN_TYPES = ("FLOW",)
    RETURN_NAMES = ("flow_upscaled",)
    FUNCTION = "refine"
    CATEGORY = "MotionTransfer/Flow"

    def refine(self, flow, guide_image, target_width, target_height, guided_filter_radius, guided_filter_eps):
        """Upscale flow fields to target resolution with edge-aware refinement.

        Args:
            flow: [B, H_lo, W_lo, 2] flow fields
            guide_image: [1, H_hi, W_hi, C] high-res still image
            target_width, target_height: Target resolution
            guided_filter_radius: Radius for edge-aware filtering
            guided_filter_eps: Regularization for guided filter

        Returns:
            flow_upscaled: [B, H_hi, W_hi, 2] upscaled and refined flow
        """
        # Convert tensors to numpy arrays properly
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        elif not isinstance(flow, np.ndarray):
            flow = np.array(flow)

        if isinstance(guide_image, torch.Tensor):
            guide_image = guide_image.cpu().numpy()
        elif not isinstance(guide_image, np.ndarray):
            guide_image = np.array(guide_image)

        # Get guide image (use first frame if batch)
        guide = guide_image[0] if len(guide_image.shape) == 4 else guide_image

        # Resize guide to target if needed
        guide_h, guide_w = guide.shape[:2]
        if guide_h != target_height or guide_w != target_width:
            guide = cv2.resize(guide, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Convert guide to grayscale for filtering
        if guide.shape[2] == 3:
            guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide[:, :, 0]

        # Upscale each flow field in batch
        flow_batch = flow.shape[0]
        flow_h, flow_w = flow.shape[1:3]
        scale_x = target_width / flow_w
        scale_y = target_height / flow_h

        upscaled_flows = []
        for i in range(flow_batch):
            flow_frame = flow[i]  # [H, W, 2]

            # Bicubic upscale with proper flow scaling
            flow_u = cv2.resize(flow_frame[:, :, 0], (target_width, target_height),
                              interpolation=cv2.INTER_CUBIC) * scale_x
            flow_v = cv2.resize(flow_frame[:, :, 1], (target_width, target_height),
                              interpolation=cv2.INTER_CUBIC) * scale_y

            # Apply guided filter if available
            if FlowSRRefine._guided_filter_available:
                flow_u_ref = cv2.ximgproc.guidedFilter(
                    guide_gray, flow_u.astype(np.float32),
                    radius=guided_filter_radius, eps=guided_filter_eps
                )
                flow_v_ref = cv2.ximgproc.guidedFilter(
                    guide_gray, flow_v.astype(np.float32),
                    radius=guided_filter_radius, eps=guided_filter_eps
                )
            else:
                if not FlowSRRefine._guided_filter_warning_shown:
                    print("WARNING: opencv-contrib-python not found, using bilateral filter instead of guided filter")
                    FlowSRRefine._guided_filter_warning_shown = True
                flow_u_ref = cv2.bilateralFilter(flow_u, guided_filter_radius, 50, 50)
                flow_v_ref = cv2.bilateralFilter(flow_v, guided_filter_radius, 50, 50)

            # Stack channels
            flow_refined = np.stack([flow_u_ref, flow_v_ref], axis=-1)
            upscaled_flows.append(flow_refined)

        result = np.stack(upscaled_flows, axis=0)  # [B, H_hi, W_hi, 2]
        return (result,)


# ------------------------------------------------------
# Node 4: FlowToSTMap - Convert flow to STMap for warping
# ------------------------------------------------------
class FlowToSTMap:
    """Convert optical flow (u,v) displacement fields into normalized STMap coordinates.

    STMap format: RG channels contain normalized UV coordinates [0,1] for texture lookup.
    Compatible with Nuke STMap node, After Effects RE:Map, and ComfyUI remap nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW", {
                    "tooltip": "High-resolution flow fields from FlowSRRefine. Will be converted to normalized STMap coordinates for warping."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stmap",)
    FUNCTION = "to_stmap"
    CATEGORY = "MotionTransfer/Flow"

    def to_stmap(self, flow):
        """Convert flow displacement to normalized STMap coordinates.

        Args:
            flow: [B, H, W, 2] flow fields containing (u, v) pixel displacements
                  IMPORTANT: flow[i] represents motion from frame i to frame i+1
                  For motion transfer, we need to ACCUMULATE flow to get total
                  displacement from the original still image.

        Returns:
            stmap: [B, H, W, 3] STMap with RG=normalized coords, B=unused (set to 0)
                   Format: S = (x + accumulated_u) / W, T = (y + accumulated_v) / H
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        batch_size, height, width, _ = flow.shape

        # Create base coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)

        # Accumulate flow vectors for motion transfer
        # flow[0] = frame0→frame1, flow[1] = frame1→frame2, etc.
        # For motion transfer from still image:
        # - Frame 0: no displacement (identity)
        # - Frame 1: flow[0]
        # - Frame 2: flow[0] + flow[1]
        # - Frame 3: flow[0] + flow[1] + flow[2]
        accumulated_flow_u = np.zeros((height, width), dtype=np.float32)
        accumulated_flow_v = np.zeros((height, width), dtype=np.float32)

        stmaps = []
        for i in range(batch_size):
            # Accumulate current flow onto total displacement
            accumulated_flow_u += flow[i, :, :, 0]
            accumulated_flow_v += flow[i, :, :, 1]

            # Compute absolute coordinates after accumulated displacement
            new_x = x_coords + accumulated_flow_u
            new_y = y_coords + accumulated_flow_v

            # Normalize to [0, 1] range for STMap
            s = new_x / (width - 1)  # Normalized S coordinate
            t = new_y / (height - 1)  # Normalized T coordinate

            # Create 3-channel STMap (RG=coords, B=unused)
            stmap = np.zeros((height, width, 3), dtype=np.float32)
            stmap[:, :, 0] = s  # R channel = S (horizontal)
            stmap[:, :, 1] = t  # G channel = T (vertical)
            stmap[:, :, 2] = 0.0  # B channel = unused

            stmaps.append(stmap)

        result = np.stack(stmaps, axis=0)  # [B, H, W, 3]
        return (result,)


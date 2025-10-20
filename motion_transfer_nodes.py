"""ComfyUI Motion Transfer Node Pack
Transfer motion from a low-res AI video to a 16K still image using flow-based warping.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ------------------------------------------------------
# Node 1: VideoFramesLoader - REMOVED, using stock ComfyUI video loader
# ------------------------------------------------------

# ------------------------------------------------------
# Node 2: RAFTFlowExtractor - Extract optical flow using RAFT
# ------------------------------------------------------
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
                    "default": 8,
                    "min": 6,
                    "max": 32,
                    "tooltip": "Refinement iterations. SEA-RAFT needs fewer (6-8) than RAFT (12-20) for same quality. Default 8 works well for both."
                }),
                "model_name": ([
                    "sea-raft-small",
                    "sea-raft-medium",
                    "sea-raft-large",
                    "raft-things",
                    "raft-sintel",
                    "raft-small"
                ], {
                    "default": "sea-raft-medium",
                    "tooltip": "Optical flow model. SEA-RAFT (recommended): 2.3x faster, 22% more accurate (ECCV 2024 Best Paper Candidate). RAFT: original (2020). 'sea-raft-medium': best speed/quality balance for 12-24GB VRAM. 'sea-raft-small': faster for 8GB VRAM. 'sea-raft-large': best quality for 24GB+ VRAM."
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

        Returns:
            tuple: (model, model_type) where model_type is 'raft' or 'searaft'
        """
        if cls._model is None or cls._model_path != model_name:
            # Detect model type
            is_searaft = model_name.startswith("sea-raft")

            if is_searaft:
                # ========== Load SEA-RAFT ==========
                try:
                    import sys
                    import os

                    # Try to find SEA-RAFT in common locations
                    searaft_paths = [
                        # If cloned to ComfyUI/custom_nodes/SEA-RAFT/core
                        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SEA-RAFT', 'core'),
                        # If cloned elsewhere and added to PYTHONPATH
                        'SEA-RAFT/core',
                    ]

                    # Try importing from each possible location
                    SEARAFT = None
                    for path in searaft_paths:
                        if os.path.exists(path) and path not in sys.path:
                            sys.path.insert(0, path)
                            try:
                                from raft import RAFT as SEARAFT
                                print(f"✓ Found SEA-RAFT at: {path}")
                                break
                            except ImportError:
                                sys.path.remove(path)

                    # If still not found, try direct import (in case it's already in PYTHONPATH)
                    if SEARAFT is None:
                        try:
                            from core.raft import RAFT as SEARAFT
                            print("✓ Found SEA-RAFT in PYTHONPATH")
                        except ImportError:
                            from raft import RAFT as SEARAFT
                            print("✓ Found SEA-RAFT in system path")

                    from huggingface_hub import hf_hub_download

                except ImportError as e:
                    raise ImportError(
                        f"SEA-RAFT not found. Please install:\n\n"
                        f"1. Clone SEA-RAFT repository:\n"
                        f"   cd ComfyUI/custom_nodes\n"
                        f"   git clone https://github.com/princeton-vl/SEA-RAFT.git\n\n"
                        f"2. Install huggingface-hub:\n"
                        f"   pip install huggingface-hub>=0.20.0\n\n"
                        f"3. Restart ComfyUI\n\n"
                        f"Error details: {e}"
                    )

                # Map model names to HuggingFace repos
                hf_models = {
                    "sea-raft-small": "MemorySlices/SEA-RAFT-S",
                    "sea-raft-medium": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                    "sea-raft-large": "MemorySlices/SEA-RAFT-L",
                }

                if model_name not in hf_models:
                    raise ValueError(f"Unknown SEA-RAFT model: {model_name}")

                repo_id = hf_models[model_name]
                print(f"Loading SEA-RAFT from HuggingFace: {repo_id}")
                print("First run will download model (~100-200MB), subsequent runs use cache...")

                try:
                    # Download checkpoint from HuggingFace Hub (auto-caches)
                    checkpoint_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="model.pth",
                        cache_dir=None  # Uses default ~/.cache/huggingface
                    )

                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)

                    # Create SEA-RAFT model (use same API as RAFT)
                    import argparse
                    args = argparse.Namespace()
                    args.small = (model_name == "sea-raft-small")
                    args.mixed_precision = False
                    args.alternate_corr = False

                    cls._model = SEARAFT(args)
                    cls._model.load_state_dict(checkpoint)
                    cls._model = cls._model.to(device).eval()
                    cls._model_path = model_name
                    cls._model_type = 'searaft'

                    print(f"✓ SEA-RAFT model loaded successfully: {model_name}")

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load SEA-RAFT model from HuggingFace.\n"
                        f"Model: {repo_id}\n"
                        f"Error: {e}\n"
                        f"Try: Check internet connection or use RAFT models instead."
                    )

            else:
                # ========== Load Original RAFT ==========
                try:
                    import sys
                    sys.path.append('path/to/RAFT/core')  # Add RAFT to path if needed
                    from raft import RAFT
                    import argparse
                except ImportError:
                    raise ImportError(
                        "RAFT not found. Install with:\n"
                        "pip install git+https://github.com/princeton-vl/RAFT.git\n"
                        "Or clone and add to PYTHONPATH"
                    )

                # Create RAFT model with default args
                args = argparse.Namespace()
                args.small = (model_name == "raft-small")
                args.mixed_precision = False
                args.alternate_corr = False

                cls._model = RAFT(args)

                # Load weights from checkpoint file
                # User must download weights and place in models/raft/ folder
                model_paths = {
                    "raft-things": "models/raft/raft-things.pth",
                    "raft-sintel": "models/raft/raft-sintel.pth",
                    "raft-small": "models/raft/raft-small.pth",
                }

                checkpoint_path = model_paths.get(model_name, model_paths["raft-sintel"])

                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    cls._model.load_state_dict(checkpoint)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"RAFT checkpoint not found at {checkpoint_path}\n"
                        f"Download from https://github.com/princeton-vl/RAFT and place in models/raft/"
                    )

                cls._model = cls._model.to(device).eval()
                cls._model_path = model_name
                cls._model_type = 'raft'

        return cls._model, cls._model_type

NODE_CLASS_MAPPINGS["RAFTFlowExtractor"] = RAFTFlowExtractor
NODE_DISPLAY_NAME_MAPPINGS["RAFTFlowExtractor"] = "RAFT Flow Extractor"

# ------------------------------------------------------
# Node 3: FlowSRRefine - Upscale and refine flow fields
# ------------------------------------------------------
class FlowSRRefine:
    """Upscale and refine optical flow fields using bicubic interpolation and guided filtering.

    Upscales low-resolution flow to match high-resolution still image, with edge-aware
    smoothing to prevent flow bleeding across sharp boundaries.
    """

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
        if isinstance(flow, np.ndarray):
            flow = flow
        else:
            flow = np.array(flow)

        if isinstance(guide_image, torch.Tensor):
            guide_image = guide_image.cpu().numpy()

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
            try:
                flow_u_ref = cv2.ximgproc.guidedFilter(
                    guide_gray, flow_u.astype(np.float32),
                    radius=guided_filter_radius, eps=guided_filter_eps
                )
                flow_v_ref = cv2.ximgproc.guidedFilter(
                    guide_gray, flow_v.astype(np.float32),
                    radius=guided_filter_radius, eps=guided_filter_eps
                )
            except AttributeError:
                # Fallback to bilateral filter if ximgproc not available
                print("WARNING: opencv-contrib-python not found, using bilateral filter instead of guided filter")
                flow_u_ref = cv2.bilateralFilter(flow_u, guided_filter_radius, 50, 50)
                flow_v_ref = cv2.bilateralFilter(flow_v, guided_filter_radius, 50, 50)

            # Stack channels
            flow_refined = np.stack([flow_u_ref, flow_v_ref], axis=-1)
            upscaled_flows.append(flow_refined)

        result = np.stack(upscaled_flows, axis=0)  # [B, H_hi, W_hi, 2]
        return (result,)

NODE_CLASS_MAPPINGS["FlowSRRefine"] = FlowSRRefine
NODE_DISPLAY_NAME_MAPPINGS["FlowSRRefine"] = "Flow SR Refine"

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

        Returns:
            stmap: [B, H, W, 3] STMap with RG=normalized coords, B=unused (set to 0)
                   Format: S = (x + u) / W, T = (y + v) / H
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        batch_size, height, width, _ = flow.shape

        # Create base coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)

        stmaps = []
        for i in range(batch_size):
            flow_frame = flow[i]  # [H, W, 2]
            flow_u = flow_frame[:, :, 0]  # Horizontal displacement
            flow_v = flow_frame[:, :, 1]  # Vertical displacement

            # Compute absolute coordinates after displacement
            new_x = x_coords + flow_u
            new_y = y_coords + flow_v

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

NODE_CLASS_MAPPINGS["FlowToSTMap"] = FlowToSTMap
NODE_DISPLAY_NAME_MAPPINGS["FlowToSTMap"] = "Flow to STMap"

# ------------------------------------------------------
# Node 5: TileWarp16K - Tiled warping for ultra-high-res images
# ------------------------------------------------------
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
        if isinstance(still_image, torch.Tensor):
            still_image = still_image.cpu().numpy()
        if isinstance(stmap, torch.Tensor):
            stmap = stmap.cpu().numpy()

        # Get still image (first frame if batch)
        still = still_image[0] if len(still_image.shape) == 4 else still_image
        h, w, c = still.shape

        # Get interpolation mode
        interp_map = {
            "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        interp_mode = interp_map[interpolation]

        # Create feather weights for overlap blending
        feather_weights = self._create_feather_mask(tile_size, overlap)

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
                    still_tile = still[y0:y1, x0:x1]
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

NODE_CLASS_MAPPINGS["TileWarp16K"] = TileWarp16K
NODE_DISPLAY_NAME_MAPPINGS["TileWarp16K"] = "Tile Warp 16K"

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
        h, w = frames.shape[1:3]

        stabilized = [frames[0]]  # First frame unchanged

        for t in range(1, batch_size):
            current_frame = frames[t]
            prev_stabilized = stabilized[-1]
            flow_fwd = flow[t-1]  # Flow from t-1 to t

            # Warp previous stabilized frame forward
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow_fwd[:, :, 0]).astype(np.float32)
            map_y = (map_y + flow_fwd[:, :, 1]).astype(np.float32)

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

NODE_CLASS_MAPPINGS["TemporalConsistency"] = TemporalConsistency
NODE_DISPLAY_NAME_MAPPINGS["TemporalConsistency"] = "Temporal Consistency"

# ------------------------------------------------------
# Node 7: HiResWriter - Export high-res sequences
# ------------------------------------------------------
class HiResWriter:
    """Write high-resolution image sequences to disk (EXR, PNG, or video).

    Supports frame sequences and video encoding with various formats.
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

NODE_CLASS_MAPPINGS["HiResWriter"] = HiResWriter
NODE_DISPLAY_NAME_MAPPINGS["HiResWriter"] = "Hi-Res Writer"

# ======================================================
# PIPELINE B - MESH-WARP NODES
# ======================================================

# ------------------------------------------------------
# Node 8: MeshBuilder2D - Build 2D mesh from flow
# ------------------------------------------------------
class MeshBuilder2D:
    """Build a 2D deformation mesh from optical flow using Delaunay triangulation.

    Creates a coarse mesh that tracks with the flow, useful for more stable
    deformation than raw pixel-based warping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW", {
                    "tooltip": "Optical flow fields from RAFTFlowExtractor. Flow will be sampled at mesh vertices to create deformation mesh."
                }),
                "mesh_resolution": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "tooltip": "Number of mesh control points along each axis. Higher values (64-128) give finer deformation control but slower. Lower values (16-32) are faster. 32 is a good balance."
                }),
                "min_triangle_area": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 10000.0,
                    "tooltip": "Minimum area for triangles (in pixels²). Filters out degenerate/tiny triangles that can cause artifacts. Lower values keep more triangles but may have issues. 100.0 is recommended."
                }),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "build_mesh"
    CATEGORY = "MotionTransfer/Mesh"

    def build_mesh(self, flow, mesh_resolution, min_triangle_area):
        """Build mesh sequence from flow fields.

        Args:
            flow: [B, H, W, 2] flow displacement fields
            mesh_resolution: Number of mesh vertices along each axis
            min_triangle_area: Minimum triangle area for filtering

        Returns:
            mesh_sequence: List of mesh dicts containing vertices, faces, uvs
        """
        from scipy.spatial import Delaunay

        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        batch_size, height, width = flow.shape[:3]

        meshes = []
        for i in range(batch_size):
            flow_frame = flow[i]

            # Create uniform grid of control points with minimum step size of 1
            step_y = max(1, height // mesh_resolution)
            step_x = max(1, width // mesh_resolution)

            vertices = []
            uvs = []

            for y in range(0, height, step_y):
                for x in range(0, width, step_x):
                    # Sample flow at this point
                    if y < height and x < width:
                        flow_u = flow_frame[y, x, 0]
                        flow_v = flow_frame[y, x, 1]

                        # Deformed vertex position
                        vert_x = x + flow_u
                        vert_y = y + flow_v

                        vertices.append([vert_x, vert_y])
                        uvs.append([x / width, y / height])

            vertices = np.array(vertices, dtype=np.float32)
            uvs = np.array(uvs, dtype=np.float32)

            # Delaunay triangulation
            tri = Delaunay(uvs)
            faces = tri.simplices

            # Filter small triangles
            valid_faces = []
            for face in faces:
                v0, v1, v2 = vertices[face]
                area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                                (v2[0] - v0[0]) * (v1[1] - v0[1]))
                if area >= min_triangle_area:
                    valid_faces.append(face)

            faces = np.array(valid_faces, dtype=np.int32)

            mesh = {
                'vertices': vertices,
                'faces': faces,
                'uvs': uvs,
                'width': width,
                'height': height,
            }
            meshes.append(mesh)

        return (meshes,)

NODE_CLASS_MAPPINGS["MeshBuilder2D"] = MeshBuilder2D
NODE_DISPLAY_NAME_MAPPINGS["MeshBuilder2D"] = "Mesh Builder 2D"

# ------------------------------------------------------
# Node 9: AdaptiveTessellate - Adaptive mesh refinement
# ------------------------------------------------------
class AdaptiveTessellate:
    """Adaptively refine mesh based on flow gradient magnitude.

    Subdivides triangles in high-motion areas for better deformation accuracy.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH", {
                    "tooltip": "Mesh sequence from MeshBuilder2D to be refined with adaptive subdivision."
                }),
                "flow": ("FLOW", {
                    "tooltip": "Flow fields used to compute gradient magnitude for adaptive subdivision. Areas with high flow gradients get more subdivision."
                }),
                "subdivision_threshold": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 100.0,
                    "tooltip": "Flow gradient threshold for triggering subdivision. Lower values (5.0) subdivide more aggressively, higher values (20.0) subdivide less. Currently placeholder - full subdivision not yet implemented."
                }),
                "max_subdivisions": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 4,
                    "tooltip": "Maximum subdivision iterations. Higher values create finer meshes but slower processing. 0 = no subdivision, 2 = balanced, 4 = very detailed. Currently placeholder."
                }),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "tessellate"
    CATEGORY = "MotionTransfer/Mesh"

    def tessellate(self, mesh_sequence, flow, subdivision_threshold, max_subdivisions):
        """Adaptively subdivide mesh based on flow gradients.

        Args:
            mesh_sequence: List of mesh dicts
            flow: [B, H, W, 2] flow fields
            subdivision_threshold: Flow gradient threshold for subdivision
            max_subdivisions: Maximum subdivision iterations

        Returns:
            refined_mesh: List of refined mesh dicts
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        refined_meshes = []
        for mesh_idx, mesh in enumerate(mesh_sequence):
            flow_frame = flow[mesh_idx]

            # Compute flow gradient magnitude
            flow_u = flow_frame[:, :, 0]
            flow_v = flow_frame[:, :, 1]

            grad_u_x = cv2.Sobel(flow_u, cv2.CV_32F, 1, 0, ksize=3)
            grad_u_y = cv2.Sobel(flow_u, cv2.CV_32F, 0, 1, ksize=3)
            grad_v_x = cv2.Sobel(flow_v, cv2.CV_32F, 1, 0, ksize=3)
            grad_v_y = cv2.Sobel(flow_v, cv2.CV_32F, 0, 1, ksize=3)

            grad_mag = np.sqrt(grad_u_x**2 + grad_u_y**2 + grad_v_x**2 + grad_v_y**2)

            # For now, return original mesh (full adaptive subdivision is complex)
            # In production, would implement Loop or Catmull-Clark subdivision
            refined_mesh = mesh.copy()

            refined_meshes.append(refined_mesh)

        return (refined_meshes,)

NODE_CLASS_MAPPINGS["AdaptiveTessellate"] = AdaptiveTessellate
NODE_DISPLAY_NAME_MAPPINGS["AdaptiveTessellate"] = "Adaptive Tessellate"

# ------------------------------------------------------
# Node 10: BarycentricWarp - Mesh-based warping
# ------------------------------------------------------
class BarycentricWarp:
    """Warp image using barycentric interpolation on triangulated mesh.

    More stable than pixel-based flow warping, especially for large deformations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "still_image": ("IMAGE", {
                    "tooltip": "High-resolution still image to warp using mesh deformation. Alternative to TileWarp16K - better for large deformations."
                }),
                "mesh_sequence": ("MESH", {
                    "tooltip": "Mesh sequence from AdaptiveTessellate (or MeshBuilder2D). Contains deformed triangles that define the warping."
                }),
                "interpolation": (["linear", "cubic"], {
                    "default": "linear",
                    "tooltip": "Interpolation method for triangle warping. 'linear': faster (recommended for mesh). 'cubic': higher quality but slower and may cause artifacts with meshes."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_sequence",)
    FUNCTION = "warp"
    CATEGORY = "MotionTransfer/Mesh"

    def warp(self, still_image, mesh_sequence, interpolation):
        """Warp image using mesh deformation.

        Args:
            still_image: [1, H, W, C] source image
            mesh_sequence: List of deformed meshes
            interpolation: Interpolation method

        Returns:
            warped_sequence: [B, H, W, C] warped frames
        """
        if isinstance(still_image, torch.Tensor):
            still_image = still_image.cpu().numpy()

        still = still_image[0] if len(still_image.shape) == 4 else still_image
        h, w, c = still.shape

        # Map interpolation string to OpenCV constant
        interp_map = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC
        }
        interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

        warped_frames = []

        for mesh in mesh_sequence:
            vertices = mesh['vertices']
            faces = mesh['faces']
            uvs = mesh['uvs']

            # Create output image
            warped = np.zeros((h, w, c), dtype=np.float32)

            # Rasterize each triangle
            for face in faces:
                # Get triangle vertices in deformed space
                v0, v1, v2 = vertices[face]

                # Get corresponding UV coordinates
                uv0, uv1, uv2 = uvs[face]

                # Convert UVs to pixel coordinates in source image
                src_v0 = [uv0[0] * w, uv0[1] * h]
                src_v1 = [uv1[0] * w, uv1[1] * h]
                src_v2 = [uv2[0] * w, uv2[1] * h]

                # Rasterize triangle with user-specified interpolation
                self._rasterize_triangle(
                    still, warped,
                    np.array([v0, v1, v2], dtype=np.float32),
                    np.array([src_v0, src_v1, src_v2], dtype=np.float32),
                    interp_flag
                )

            warped_frames.append(warped)

        result = np.stack(warped_frames, axis=0)
        return (result,)

    def _rasterize_triangle(self, src_image, dst_image, dst_tri, src_tri, interp_flag):
        """Rasterize a single triangle using affine transformation.

        Args:
            src_image: Source image
            dst_image: Destination image (modified in-place)
            dst_tri: Destination triangle vertices (3x2)
            src_tri: Source triangle vertices (3x2)
            interp_flag: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR)
        """
        # Get bounding box
        x_min = int(max(0, np.floor(dst_tri[:, 0].min())))
        x_max = int(min(dst_image.shape[1], np.ceil(dst_tri[:, 0].max())))
        y_min = int(max(0, np.floor(dst_tri[:, 1].min())))
        y_max = int(min(dst_image.shape[0], np.ceil(dst_tri[:, 1].max())))

        if x_max <= x_min or y_max <= y_min:
            return

        # Use OpenCV's warpAffine for triangle
        # Get affine transform
        try:
            M = cv2.getAffineTransform(dst_tri.astype(np.float32), src_tri.astype(np.float32))

            # Create mask for triangle
            mask = np.zeros(dst_image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), 255)

            # Warp region with user-specified interpolation
            warped_region = cv2.warpAffine(
                src_image,
                M,
                (dst_image.shape[1], dst_image.shape[0]),
                flags=interp_flag,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Blend using mask
            mask_3ch = (mask[:, :, None] / 255.0).astype(np.float32)
            dst_image[:] = dst_image * (1 - mask_3ch) + warped_region * mask_3ch

        except cv2.error:
            # Skip degenerate triangles
            pass

NODE_CLASS_MAPPINGS["BarycentricWarp"] = BarycentricWarp
NODE_DISPLAY_NAME_MAPPINGS["BarycentricWarp"] = "Barycentric Warp"

# ======================================================
# PIPELINE C - 3D-PROXY NODES (Depth-based warping)
# ======================================================

# ------------------------------------------------------
# Node 11: DepthEstimator - Monocular depth estimation
# ------------------------------------------------------
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

NODE_CLASS_MAPPINGS["DepthEstimator"] = DepthEstimator
NODE_DISPLAY_NAME_MAPPINGS["DepthEstimator"] = "Depth Estimator"

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
            depth_maps: [B, H, W, 1] depth sequence
            flow: [B, H, W, 2] optical flow for motion estimation
            focal_length: Camera focal length estimate

        Returns:
            reprojected_sequence: [B, H, W, C] reprojected frames
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

NODE_CLASS_MAPPINGS["ProxyReprojector"] = ProxyReprojector
NODE_DISPLAY_NAME_MAPPINGS["ProxyReprojector"] = "Proxy Reprojector"

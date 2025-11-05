"""
Combined sequential motion transfer node.

Provides a single all-in-one node that chains the entire Pipeline A workflow
for simplified usage.
"""

import torch
import numpy as np
from typing import Tuple

# Import required nodes from other modules
from .flow_nodes import RAFTFlowExtractor, FlowSRRefine, FlowToSTMap
from .warp_nodes import TileWarp16K, TemporalConsistency, HiResWriter


class SequentialMotionTransfer:
    """End-to-end motion transfer pipeline that processes frames sequentially.

    Runs RAFT/SEA-RAFT, flow refinement, STMap conversion, warping, temporal
    stabilization, and writing in a single pass so only one high-resolution
    frame is resident in memory at any time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames from the VHS loader. Entire clip will be processed sequentially."
                }),
                "still_image": ("IMAGE", {
                    "tooltip": "High-resolution still image to warp (e.g. 16K artwork)."
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
                    "tooltip": "Optical flow backbone. SEA-RAFT models auto-download (HuggingFace) and are faster."
                }),
                "raft_iters": ("INT", {
                    "default": 12,
                    "min": 4,
                    "max": 32,
                    "tooltip": "Refinement iterations for RAFT. Automatically reduced for SEA-RAFT if left at default."
                }),
                "target_width": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target width for refined flow / warped output. Should match still image."
                }),
                "target_height": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target height for refined flow / warped output. Should match still image."
                }),
                "guided_filter_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Guided/bilateral filter radius for flow refinement."
                }),
                "guided_filter_eps": ("FLOAT", {
                    "default": 1e-3,
                    "min": 1e-6,
                    "max": 1.0,
                    "tooltip": "Guided filter regularization epsilon."
                }),
                "tile_size": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "tooltip": "Warping tile size. Reduce to lower VRAM usage."
                }),
                "overlap": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 512,
                    "tooltip": "Tile overlap to feather seams."
                }),
                "interpolation": (["cubic", "linear", "lanczos4"], {
                    "default": "cubic",
                    "tooltip": "Interpolation kernel for warping."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Temporal blend weight. Higher = smoother, lower = sharper."
                }),
                "output_path": ("STRING", {
                    "default": "output/frame",
                    "tooltip": "Output file prefix. Frames are written as prefix_0000.ext, prefix_0001.ext, ..."
                }),
                "format": (["png", "exr", "jpg"], {
                    "default": "png",
                    "tooltip": "Output image format."
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame number for filenames."
                }),
                "return_sequence": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reload written frames and return them as IMAGE output (uses memory proportional to clip length)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MotionTransfer/Pipeline"

    def run(
        self,
        images,
        still_image,
        model_name,
        raft_iters,
        target_width,
        target_height,
        guided_filter_radius,
        guided_filter_eps,
        tile_size,
        overlap,
        interpolation,
        blend_strength,
        output_path,
        format,
        start_frame,
        return_sequence,
    ):
        from pathlib import Path
        from torch.nn.functional import pad

        if isinstance(images, torch.Tensor):
            images_tensor = images.detach().cpu()
        else:
            images_tensor = torch.from_numpy(np.asarray(images))

        if images_tensor.ndim != 4 or images_tensor.shape[-1] != 3:
            raise ValueError("Expected video frames shaped [B, H, W, 3]")

        batch_size = images_tensor.shape[0]
        if batch_size < 2:
            raise ValueError("Need at least 2 video frames to compute optical flow.")

        # Convert to BCHW float tensor (keep on CPU, move per-frame to device)
        video_bchw = images_tensor.permute(0, 3, 1, 2).contiguous().float()

        if isinstance(still_image, torch.Tensor):
            still_np = still_image.detach().cpu().numpy()
        else:
            still_np = np.asarray(still_image)

        if still_np.ndim == 4:
            still_np = still_np[0]
        if still_np.ndim != 3 or still_np.shape[2] != 3:
            raise ValueError("Still image must be [H, W, 3]")

        still_np = still_np.astype(np.float32)

        # Resize still to target if needed
        if still_np.shape[1] != target_width or still_np.shape[0] != target_height:
            still_np = cv2.resize(still_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        still_batch = np.expand_dims(still_np, axis=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_type = OpticalFlowModel.load(model_name, device)

        # Auto-adjust iterations for SEA-RAFT default
        effective_iters = raft_iters
        if model_type == "searaft" and raft_iters == 12:
            effective_iters = OpticalFlowModel.get_recommended_iters("searaft")
            print(f"[Sequential Motion Transfer] Auto-adjusted iterations to {effective_iters} for SEA-RAFT")

        # Prepare helper node instances for reuse
        flow_refiner = FlowSRRefine()
        stmap_node = FlowToSTMap()
        warp_node = TileWarp16K()
        writer_node = HiResWriter()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        prev_stabilized = None
        current_frame_idx = start_frame
        coord_cache = None  # Cache pixel grids for temporal warp
        written_files = []

        # CRITICAL FIX: Accumulate flow for motion transfer
        # Each frame needs TOTAL displacement from original still, not just frame-to-frame
        accumulated_flow_u = None
        accumulated_flow_v = None

        print(f"[Sequential Motion Transfer] Processing {batch_size-1} frames sequentially...")

        for t in range(batch_size - 1):
            img1 = video_bchw[t:t+1].to(device) * 255.0
            img2 = video_bchw[t+1:t+2].to(device) * 255.0

            h, w = img1.shape[2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img1 = pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
                img2 = pad(img2, (0, pad_w, 0, pad_h), mode="replicate")

            with torch.no_grad():
                output = model(img1, img2, iters=effective_iters, test_mode=True)

            if isinstance(output, dict):
                flow_tensor = output.get("final")
                if flow_tensor is None:
                    raise RuntimeError("SEA-RAFT output missing 'final' flow prediction.")
            elif isinstance(output, (tuple, list)):
                # Support both 2-tuple and 3-tuple variants
                flow_tensor = output[-1]
            else:
                raise RuntimeError(f"Unexpected RAFT output type: {type(output)}")

            if pad_h > 0 or pad_w > 0:
                flow_tensor = flow_tensor[:, :, :h, :w]

            flow_np = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()

            refined_flow_batch = flow_refiner.refine(
                np.expand_dims(flow_np, axis=0),
                still_batch,
                target_width,
                target_height,
                guided_filter_radius,
                guided_filter_eps,
            )[0]
            refined_flow = refined_flow_batch[0]  # [H_hi, W_hi, 2]

            # CRITICAL FIX: Accumulate flow instead of using frame-to-frame flow
            # Motion transfer needs total displacement from original still image
            if accumulated_flow_u is None:
                accumulated_flow_u = np.zeros((target_height, target_width), dtype=np.float32)
                accumulated_flow_v = np.zeros((target_height, target_width), dtype=np.float32)

            accumulated_flow_u += refined_flow[:, :, 0]
            accumulated_flow_v += refined_flow[:, :, 1]

            # Build STMap directly from accumulated flow (bypass FlowToSTMap to avoid double accumulation)
            y_coords, x_coords = np.mgrid[0:target_height, 0:target_width].astype(np.float32)
            new_x = x_coords + accumulated_flow_u
            new_y = y_coords + accumulated_flow_v

            # Normalize to [0, 1] range for STMap
            s = new_x / (target_width - 1)
            t = new_y / (target_height - 1)

            # Create 3-channel STMap (RG=coords, B=unused)
            stmap_frame = np.zeros((target_height, target_width, 3), dtype=np.float32)
            stmap_frame[:, :, 0] = s
            stmap_frame[:, :, 1] = t
            stmap_frame[:, :, 2] = 0.0

            warped_batch = warp_node.warp(
                still_batch,
                np.expand_dims(stmap_frame, axis=0),
                tile_size,
                overlap,
                interpolation,
            )[0]
            warped_frame = warped_batch[0]

            if prev_stabilized is None or blend_strength <= 0.0:
                stabilized = warped_frame.astype(np.float32)
            else:
                if coord_cache is None:
                    base_x, base_y = np.meshgrid(
                        np.arange(target_width, dtype=np.float32),
                        np.arange(target_height, dtype=np.float32),
                    )
                    coord_cache = (base_x, base_y)
                else:
                    base_x, base_y = coord_cache

                map_x = (base_x - refined_flow[:, :, 0]).astype(np.float32)
                map_y = (base_y - refined_flow[:, :, 1]).astype(np.float32)

                warped_prev = cv2.remap(
                    prev_stabilized.astype(np.float32),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                stabilized = cv2.addWeighted(
                    warped_frame.astype(np.float32), 1.0 - blend_strength,
                    warped_prev.astype(np.float32), blend_strength,
                    0.0,
                )

            filename = self._build_filename(output_path, current_frame_idx, format)
            self._write_frame(writer_node, stabilized, filename, format)
            written_files.append(filename)

            print(f"[Sequential Motion Transfer] Wrote frame {current_frame_idx}: {filename}")

            prev_stabilized = stabilized
            current_frame_idx += 1

            # Free GPU memory for next iteration
            del refined_flow, stmap_frame, warped_frame
            del flow_tensor, flow_np, refined_flow_batch, warped_batch, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[Sequential Motion Transfer] Completed {current_frame_idx - start_frame} frames.")

        if return_sequence and written_files:
            reloaded_frames = []
            for fname in written_files:
                frame = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    raise FileNotFoundError(f"Failed to read written frame: {fname}")

                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)

                # cv2 loads as BGR; convert to RGB for consistency
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    raise ValueError(f"Unsupported channel count when reloading frame {fname}: {frame.shape[2]}")

                if format in ("png", "jpg"):
                    frame = frame.astype(np.float32) / 255.0
                else:
                    frame = frame.astype(np.float32)

                reloaded_frames.append(frame)

            frames_out = np.stack(reloaded_frames, axis=0)
        else:
            frames_out = np.expand_dims(prev_stabilized.astype(np.float32), axis=0) if prev_stabilized is not None else np.zeros((0, target_height, target_width, 3), dtype=np.float32)

        return (frames_out,)

    def _build_filename(self, output_path: str, frame_idx: int, format: str) -> str:
        return f"{output_path}_{frame_idx:04d}.{format}"

    def _write_frame(self, writer_node: HiResWriter, image: np.ndarray, filename: str, format: str):
        # Reuse HiResWriter helpers to keep format handling consistent
        if format == "exr":
            writer_node._write_exr(image, filename)
        elif format == "png":
            writer_node._write_png(image, filename)
        elif format == "jpg":
            writer_node._write_jpg(image, filename)
        else:
            raise ValueError(f"Unsupported output format: {format}")


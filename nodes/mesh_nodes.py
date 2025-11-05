"""
Mesh generation and warping nodes.

Contains nodes for Delaunay triangulation, adaptive tessellation, CoTracker integration,
and barycentric mesh warping.
"""

import torch
import numpy as np
import cv2
import json
from scipy.spatial import Delaunay
from typing import Tuple, List, Optional, Dict

# Try to import CUDA accelerated kernels
try:
    from ..cuda import cuda_loader
    CUDA_AVAILABLE = cuda_loader.is_cuda_available()
except ImportError:
    CUDA_AVAILABLE = False

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


# ------------------------------------------------------
# Node 9B: MeshFromCoTracker - Build mesh from CoTracker trajectories
# ------------------------------------------------------
class MeshFromCoTracker:
    """Build deformation mesh from CoTracker point trajectories.

    Converts sparse point tracks [T, N, 2] from CoTracker into triangulated mesh sequence
    compatible with BarycentricWarp node. This provides better temporal stability than
    RAFT-based mesh generation, especially for large deformations and organic motion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracking_results": ("STRING", {
                    "tooltip": "JSON tracking results from CoTrackerNode. Contains point trajectories [T, N, 2] where T=frames, N=points, 2=XY coordinates."
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Which frame to use as reference (usually 0). Mesh deformation is relative to this frame's point positions."
                }),
                "min_triangle_area": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 10000.0,
                    "tooltip": "Minimum area for triangles (in pixels²). Filters out degenerate/tiny triangles that can cause artifacts. Same as MeshBuilder2D parameter. 100.0 is recommended."
                }),
                "video_width": ("INT", {
                    "default": 1920,
                    "min": 64,
                    "max": 8192,
                    "tooltip": "Original video width (for UV normalization). Should match the video used for tracking."
                }),
                "video_height": ("INT", {
                    "default": 1080,
                    "min": 64,
                    "max": 8192,
                    "tooltip": "Original video height (for UV normalization). Should match the video used for tracking."
                }),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "build_mesh_from_tracks"
    CATEGORY = "MotionTransfer/Mesh"

    def build_mesh_from_tracks(self, tracking_results, frame_index, min_triangle_area, video_width, video_height):
        """Convert CoTracker trajectories to mesh sequence.

        Args:
            tracking_results: STRING from CoTrackerNode (ComfyUI returns first item from list)
                             CoTracker outputs list of JSON strings, one per point
            frame_index: Reference frame (usually 0)
            min_triangle_area: Filter threshold for degenerate triangles
            video_width, video_height: Original video dimensions

        Returns:
            mesh_sequence: List of mesh dicts (same format as MeshBuilder2D)
        """
        import json
        from scipy.spatial import Delaunay

        # CoTrackerNode returns list of JSON strings via RETURN_TYPES = ("STRING",...)
        # ComfyUI converts list → newline-separated string when passing to next node
        # Format: "point1_json\npoint2_json\npoint3_json..."
        # Each line is: [{"x":500,"y":300}, {"x":502,"y":301}, ...]

        lines = tracking_results.strip().split('\n')
        if len(lines) == 0:
            raise ValueError("No tracking data found in tracking_results")

        # Parse each point's trajectory
        point_trajectories = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                trajectory = json.loads(line)  # List of {"x": int, "y": int}
                point_trajectories.append(trajectory)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

        if len(point_trajectories) == 0:
            raise ValueError("No valid trajectory data found in tracking_results")

        N = len(point_trajectories)  # Number of points
        T = len(point_trajectories[0])  # Number of frames

        print(f"MeshFromCoTracker: Loaded {N} points across {T} frames")

        # Convert to [T, N, 2] array
        trajectories = np.zeros((T, N, 2), dtype=np.float32)
        for point_idx, trajectory in enumerate(point_trajectories):
            for frame_idx, coord in enumerate(trajectory):
                trajectories[frame_idx, point_idx, 0] = coord['x']
                trajectories[frame_idx, point_idx, 1] = coord['y']

        # Reference frame (frame 0 or specified)
        if frame_index >= T:
            frame_index = 0
            print(f"Warning: frame_index out of range, using frame 0")

        ref_points = trajectories[frame_index]  # [N, 2]

        # Build UVs from reference positions (normalized [0, 1])
        uvs = np.array([
            [x / video_width, y / video_height]
            for x, y in ref_points
        ], dtype=np.float32)

        # Delaunay triangulation on reference frame
        print("Building Delaunay triangulation...")
        tri = Delaunay(uvs)
        faces_base = tri.simplices

        # Filter degenerate triangles
        valid_faces = []
        for face in faces_base:
            v0, v1, v2 = ref_points[face]
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                            (v2[0] - v0[0]) * (v1[1] - v0[1]))
            if area >= min_triangle_area:
                valid_faces.append(face)

        faces = np.array(valid_faces, dtype=np.int32)
        print(f"Mesh: {N} vertices, {len(faces)} triangles (filtered from {len(faces_base)})")

        # Build mesh for each frame
        meshes = []
        for t in range(T):
            vertices = trajectories[t]  # [N, 2] - deformed positions

            mesh = {
                'vertices': vertices.astype(np.float32),
                'faces': faces,
                'uvs': uvs,
                'width': video_width,
                'height': video_height
            }
            meshes.append(mesh)

        print(f"✓ Built {len(meshes)} mesh frames from CoTracker trajectories")
        return (meshes,)


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

        # Try CUDA acceleration first
        if CUDA_AVAILABLE and torch.cuda.is_available():
            try:
                return self._warp_cuda(still, mesh_sequence)
            except Exception as e:
                print(f"[BarycentricWarp] CUDA failed ({e}), falling back to CPU")

        # CPU fallback
        return self._warp_cpu(still, mesh_sequence, interpolation)

    def _warp_cuda(self, still, mesh_sequence):
        """CUDA-accelerated mesh rasterization (10-20× faster than CPU)."""
        h, w, c = still.shape

        # Initialize CUDA warper
        warp_engine = cuda_loader.CUDABarycentricWarp(still.astype(np.float32))

        warped_frames = []
        for mesh in mesh_sequence:
            vertices = mesh['vertices']  # [N, 2]
            faces = mesh['faces']  # [num_tri, 3]
            uvs = mesh['uvs']  # [N, 2]

            # Build dst/src vertex arrays for all triangles
            num_triangles = len(faces)
            dst_vertices = np.zeros((num_triangles, 3, 2), dtype=np.float32)
            src_vertices = np.zeros((num_triangles, 3, 2), dtype=np.float32)

            for tri_idx, face in enumerate(faces):
                # Deformed triangle vertices
                dst_vertices[tri_idx, 0] = vertices[face[0]]
                dst_vertices[tri_idx, 1] = vertices[face[1]]
                dst_vertices[tri_idx, 2] = vertices[face[2]]

                # Source triangle vertices (UVs converted to pixel coords)
                src_vertices[tri_idx, 0] = [uvs[face[0]][0] * w, uvs[face[0]][1] * h]
                src_vertices[tri_idx, 1] = [uvs[face[1]][0] * w, uvs[face[1]][1] * h]
                src_vertices[tri_idx, 2] = [uvs[face[2]][0] * w, uvs[face[2]][1] * h]

            # CUDA rasterization (all triangles in parallel)
            warped = warp_engine.warp_mesh(dst_vertices, src_vertices, num_triangles)
            warped_frames.append(warped[:, :, :c])  # Trim to original channels

        result = np.stack(warped_frames, axis=0)
        return (result,)

    def _warp_cpu(self, still, mesh_sequence, interpolation):
        """CPU fallback (original sequential implementation)."""
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

            # Rasterize each triangle (sequential loop - slow!)
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


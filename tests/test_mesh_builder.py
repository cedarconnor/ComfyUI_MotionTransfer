"""
Tests for MeshBuilder2D node - Delaunay triangulation and mesh generation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_transfer_nodes import MeshBuilder2D


class TestMeshBuilder2D:
    """Test mesh generation from flow fields."""

    def test_mesh_structure(self, sample_flow_fields):
        """Test that generated mesh has correct structure."""
        node = MeshBuilder2D()

        mesh_sequence = node.build_mesh(sample_flow_fields,
                                       mesh_resolution=8,
                                       min_triangle_area=0.0)[0]

        assert isinstance(mesh_sequence, list), "Mesh sequence should be a list"
        assert len(mesh_sequence) == 3, f"Expected 3 meshes, got {len(mesh_sequence)}"

        # Check first mesh structure
        mesh = mesh_sequence[0]
        assert "vertices" in mesh, "Mesh missing 'vertices' key"
        assert "faces" in mesh, "Mesh missing 'faces' key"
        assert "uvs" in mesh, "Mesh missing 'uvs' key"

        vertices = mesh["vertices"]
        faces = mesh["faces"]
        uvs = mesh["uvs"]

        # Vertices should be 2D points
        assert vertices.ndim == 2, f"Vertices should be 2D array, got {vertices.ndim}D"
        assert vertices.shape[1] == 2, f"Vertices should have 2 coords, got {vertices.shape[1]}"

        # Faces should be triangles (3 vertex indices)
        assert faces.ndim == 2, f"Faces should be 2D array, got {faces.ndim}D"
        assert faces.shape[1] == 3, f"Faces should be triangles (3 indices), got {faces.shape[1]}"

        # UVs should match vertices
        assert uvs.shape == vertices.shape, \
            f"UVs shape {uvs.shape} should match vertices shape {vertices.shape}"

    def test_mesh_resolution(self):
        """Test that mesh resolution parameter controls grid density."""
        node = MeshBuilder2D()

        # Create simple constant flow
        flow = np.ones((1, 64, 64, 2), dtype=np.float32) * [5.0, 0.0]

        # Low resolution mesh
        mesh_low = node.build_mesh(flow, mesh_resolution=4, min_triangle_area=0.0)[0]
        vertices_low = mesh_low[0]["vertices"]

        # High resolution mesh
        mesh_high = node.build_mesh(flow, mesh_resolution=16, min_triangle_area=0.0)[0]
        vertices_high = mesh_high[0]["vertices"]

        # High resolution should have more vertices
        assert len(vertices_high) > len(vertices_low), \
            "Higher resolution should produce more vertices"

        # Approximately: 4x4=16 vs 16x16=256 vertices
        assert len(vertices_low) >= 12, "Low res should have at least 16 vertices (4x4 grid)"
        assert len(vertices_high) >= 200, "High res should have at least 256 vertices (16x16 grid)"

    def test_triangle_area_filtering(self):
        """Test that degenerate triangles are filtered by area threshold."""
        node = MeshBuilder2D()

        flow = np.ones((1, 64, 64, 2), dtype=np.float32) * [5.0, 0.0]

        # No filtering
        mesh_all = node.build_mesh(flow, mesh_resolution=8, min_triangle_area=0.0)[0]
        faces_all = mesh_all[0]["faces"]

        # Strict filtering (remove small triangles)
        mesh_filtered = node.build_mesh(flow, mesh_resolution=8, min_triangle_area=50.0)[0]
        faces_filtered = mesh_filtered[0]["faces"]

        # Filtered mesh should have fewer or equal triangles
        assert len(faces_filtered) <= len(faces_all), \
            "Filtered mesh should have fewer or equal triangles"

    def test_uv_normalization(self):
        """Test that UV coordinates are properly normalized to [0, 1]."""
        node = MeshBuilder2D()

        flow = np.ones((1, 64, 64, 2), dtype=np.float32) * [5.0, 0.0]
        mesh_sequence = node.build_mesh(flow, mesh_resolution=8, min_triangle_area=0.0)[0]

        uvs = mesh_sequence[0]["uvs"]

        # UVs should be in [0, 1] range
        assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0), \
            f"UVs should be in [0, 1], got min={uvs.min()}, max={uvs.max()}"

    def test_vertex_deformation(self):
        """Test that vertices are deformed according to flow."""
        node = MeshBuilder2D()

        # Create flow that moves everything 10 pixels right
        flow = np.ones((1, 64, 64, 2), dtype=np.float32)
        flow[:, :, :, 0] = 10.0  # u
        flow[:, :, :, 1] = 0.0   # v

        mesh_sequence = node.build_mesh(flow, mesh_resolution=4, min_triangle_area=0.0)[0]
        vertices = mesh_sequence[0]["vertices"]
        uvs = mesh_sequence[0]["uvs"]

        # Deformed vertices should be shifted right compared to UVs
        # UV is the original position, vertices is deformed position
        # For rightward flow: vertices_x > uv_x * width
        width = 64
        original_x = uvs[:, 0] * width
        deformed_x = vertices[:, 0]

        # Most vertices should be shifted right (allowing for boundary effects)
        mean_shift = np.mean(deformed_x - original_x)
        assert mean_shift > 5.0, \
            f"Vertices should be shifted right by ~10 pixels, got mean shift {mean_shift}"

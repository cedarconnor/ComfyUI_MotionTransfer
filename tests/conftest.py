"""
Pytest fixtures and test configuration for Motion Transfer tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_video_frames():
    """Generate synthetic video frames for testing (4 frames, 64x64, RGB)."""
    batch_size = 4
    height, width = 64, 64
    channels = 3

    # Create simple gradient pattern that shifts across frames
    frames = []
    for i in range(batch_size):
        frame = np.zeros((height, width, channels), dtype=np.float32)
        # Horizontal gradient that shifts right each frame
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = ((x + i * 5) % width) / width
                frame[y, x, 1] = y / height
                frame[y, x, 2] = 0.5
        frames.append(frame)

    return np.stack(frames, axis=0)  # [B, H, W, C]


@pytest.fixture
def sample_flow_fields():
    """Generate synthetic flow fields (3 flows for 4 frames)."""
    batch_size = 3  # B-1 flows for B frames
    height, width = 64, 64

    flows = []
    for i in range(batch_size):
        # Create constant rightward flow (5 pixels per frame)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[:, :, 0] = 5.0  # u (horizontal)
        flow[:, :, 1] = 0.0  # v (vertical)
        flows.append(flow)

    return np.stack(flows, axis=0)  # [B-1, H, W, 2]


@pytest.fixture
def sample_still_image():
    """Generate high-resolution still image for testing (256x256)."""
    height, width = 256, 256
    channels = 3

    # Create checkerboard pattern for testing tiling and warping
    still = np.zeros((height, width, channels), dtype=np.float32)
    tile_size = 32
    for y in range(height):
        for x in range(width):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                still[y, x] = [1.0, 1.0, 1.0]  # White
            else:
                still[y, x] = [0.0, 0.0, 0.0]  # Black

    return still[np.newaxis, ...]  # [1, H, W, C]


@pytest.fixture
def sample_mesh():
    """Generate simple triangular mesh for testing."""
    # Create 3x3 grid of vertices (9 vertices total)
    vertices = []
    for y in range(3):
        for x in range(3):
            vertices.append([x * 32.0, y * 32.0])
    vertices = np.array(vertices, dtype=np.float32)  # [9, 2]

    # Create triangles (8 triangles for 3x3 grid)
    faces = [
        [0, 1, 3], [1, 4, 3],  # Top-left quad
        [1, 2, 4], [2, 5, 4],  # Top-right quad
        [3, 4, 6], [4, 7, 6],  # Bottom-left quad
        [4, 5, 7], [5, 8, 7],  # Bottom-right quad
    ]
    faces = np.array(faces, dtype=np.int32)  # [8, 3]

    # UV coordinates (normalized)
    uvs = vertices / 64.0  # Normalize to [0, 1]

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs
    }


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return str(output_dir)

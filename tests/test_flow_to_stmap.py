"""
Tests for FlowToSTMap node - critical flow accumulation logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_transfer_nodes import FlowToSTMap


class TestFlowToSTMap:
    """Test flow accumulation and STMap generation."""

    def test_flow_accumulation_simple(self, sample_flow_fields):
        """Test that flow vectors are correctly accumulated frame by frame."""
        node = FlowToSTMap()

        # Input: 3 constant flows of (5, 0) pixels
        # Expected accumulated flow:
        # Frame 0: (0, 0) - identity
        # Frame 1: (5, 0) - flow[0]
        # Frame 2: (10, 0) - flow[0] + flow[1]
        # Frame 3: (15, 0) - flow[0] + flow[1] + flow[2]

        stmap = node.convert(sample_flow_fields)
        stmap_array = stmap[0]  # Unwrap tuple

        # Should have B frames (not B-1)
        assert stmap_array.shape[0] == 4, "STMap should have same number of frames as input video"

        # Check frame 0 is identity (no displacement)
        frame0 = stmap_array[0]
        h, w = frame0.shape[:2]

        # STMap R channel should be x/width at identity
        expected_r = np.tile(np.linspace(0, 1, w), (h, 1))
        np.testing.assert_allclose(frame0[:, :, 0], expected_r, rtol=0.01,
                                   err_msg="Frame 0 should be identity mapping")

        # Check frame 1 has accumulated flow[0] = 5 pixels right
        frame1 = stmap_array[1]
        # At pixel (0, 0), STMap should point to (5, 0) -> S = 5/w
        expected_shift = 5.0 / w
        assert frame1[0, 0, 0] > expected_r[0, 0], "Frame 1 should have rightward shift"

    def test_stmap_shape(self, sample_flow_fields):
        """Test that STMap has correct dimensions."""
        node = FlowToSTMap()
        stmap = node.convert(sample_flow_fields)[0]

        batch, h, w, channels = stmap.shape

        # Should output B frames (not B-1)
        assert batch == 4, f"Expected 4 frames, got {batch}"
        assert h == 64 and w == 64, f"Expected 64x64, got {h}x{w}"
        assert channels == 3, f"Expected 3 channels (RGB/STG), got {channels}"

    def test_stmap_range(self, sample_flow_fields):
        """Test that STMap values are in valid range [0, 1]."""
        node = FlowToSTMap()
        stmap = node.convert(sample_flow_fields)[0]

        # R and G channels should be normalized to [0, 1]
        assert np.all(stmap[:, :, :, 0] >= 0.0) and np.all(stmap[:, :, :, 0] <= 1.0), \
            "R channel (S) should be in [0, 1]"
        assert np.all(stmap[:, :, :, 1] >= 0.0) and np.all(stmap[:, :, :, 1] <= 1.0), \
            "G channel (T) should be in [0, 1]"

    def test_flow_accumulation_correctness(self):
        """Test flow accumulation with varying flow vectors."""
        node = FlowToSTMap()

        # Create varying flows: flow[0] = (10, 0), flow[1] = (5, 5), flow[2] = (-5, 10)
        flows = np.array([
            np.ones((64, 64, 2), dtype=np.float32) * [10, 0],
            np.ones((64, 64, 2), dtype=np.float32) * [5, 5],
            np.ones((64, 64, 2), dtype=np.float32) * [-5, 10],
        ])

        stmap = node.convert(flows)[0]
        h, w = 64, 64

        # Frame 0: identity
        # Frame 1: (10, 0)
        # Frame 2: (10+5, 0+5) = (15, 5)
        # Frame 3: (15-5, 5+10) = (10, 15)

        # Check frame 2 at pixel (32, 32) - center
        # Expected displacement: (15, 5) pixels
        # STMap S = (32 + 15) / 64 = 47/64 ≈ 0.734
        # STMap T = (32 + 5) / 64 = 37/64 ≈ 0.578
        frame2_s = stmap[2, 32, 32, 0]
        frame2_t = stmap[2, 32, 32, 1]

        expected_s = (32 + 15) / 64.0
        expected_t = (32 + 5) / 64.0

        np.testing.assert_allclose(frame2_s, expected_s, rtol=0.05,
                                   err_msg="Frame 2 S channel incorrect")
        np.testing.assert_allclose(frame2_t, expected_t, rtol=0.05,
                                   err_msg="Frame 2 T channel incorrect")

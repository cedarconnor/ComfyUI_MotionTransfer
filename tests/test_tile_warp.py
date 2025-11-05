"""
Tests for TileWarp16K node - tiled warping with feathered blending.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_transfer_nodes import TileWarp16K


class TestTileWarp16K:
    """Test tiled warping and feather blending."""

    def test_feather_mask_generation(self):
        """Test that feather masks are correctly generated."""
        node = TileWarp16K()

        tile_h, tile_w = 128, 128
        overlap = 16

        # Test interior tile (not on any edge)
        feather = node._get_tile_feather(tile_h, tile_w, 128, overlap,
                                        is_top=False, is_left=False,
                                        is_bottom=False, is_right=False)

        assert feather.shape == (tile_h, tile_w, 1), "Feather mask shape incorrect"

        # Check that edges have gradients
        # Left edge should fade from 0 to 1
        assert feather[64, 0, 0] < 0.1, "Left edge should start near 0"
        assert feather[64, overlap-1, 0] > 0.9, "Left edge should end near 1"

        # Center should be 1.0 (full weight)
        assert feather[64, 64, 0] == 1.0, "Center should have full weight"

    def test_feather_edge_tiles(self):
        """Test that edge tiles have no feathering on outer edges."""
        node = TileWarp16K()

        tile_h, tile_w = 128, 128
        overlap = 16

        # Top-left corner tile (no feathering on top and left)
        feather = node._get_tile_feather(tile_h, tile_w, 128, overlap,
                                        is_top=True, is_left=True,
                                        is_bottom=False, is_right=False)

        # Top-left corner should have full weight (no fade-in)
        assert feather[0, 0, 0] == 1.0, "Top-left corner should have full weight"

        # Right edge should have fade-out
        assert feather[64, -1, 0] < 0.1, "Right edge should fade to 0"

    def test_warp_output_shape(self, sample_still_image):
        """Test that warped output has correct dimensions."""
        node = TileWarp16K()

        # Create simple identity STMap (no warping)
        h, w = 256, 256
        stmap_sequence = []
        for i in range(3):
            stmap = np.zeros((h, w, 3), dtype=np.float32)
            # Create identity mapping
            for y in range(h):
                for x in range(w):
                    stmap[y, x, 0] = x / w  # S
                    stmap[y, x, 1] = y / h  # T
            stmap_sequence.append(stmap)

        stmap_batch = np.stack(stmap_sequence, axis=0)

        warped = node.warp(sample_still_image, stmap_batch,
                          tile_size=128, overlap=16, interpolation="linear")[0]

        assert warped.shape == (3, 256, 256, 3), \
            f"Expected (3, 256, 256, 3), got {warped.shape}"

    def test_identity_warp(self, sample_still_image):
        """Test that identity STMap produces unchanged output."""
        node = TileWarp16K()

        still = sample_still_image[0]  # Remove batch dimension
        h, w = still.shape[:2]

        # Create identity STMap
        stmap = np.zeros((1, h, w, 3), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                stmap[0, y, x, 0] = x / w
                stmap[0, y, x, 1] = y / h

        warped = node.warp(sample_still_image, stmap,
                          tile_size=128, overlap=16, interpolation="linear")[0]

        # Output should be very similar to input (allowing for interpolation errors)
        warped_frame = warped[0]
        np.testing.assert_allclose(warped_frame, still, rtol=0.01, atol=0.01,
                                   err_msg="Identity warp should preserve image")

    def test_tiling_seamless(self, sample_still_image):
        """Test that tiling produces seamless results (no visible seams)."""
        node = TileWarp16K()

        h, w = 256, 256

        # Create identity STMap
        stmap = np.zeros((1, h, w, 3), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                stmap[0, y, x, 0] = x / w
                stmap[0, y, x, 1] = y / h

        # Use small tiles to ensure multiple tiles are processed
        warped = node.warp(sample_still_image, stmap,
                          tile_size=64, overlap=16, interpolation="linear")[0]

        # Check that there are no abrupt discontinuities at tile boundaries
        # Tile boundaries are at x=64, 128, 192 (accounting for overlap)
        warped_frame = warped[0]

        # Check horizontal continuity at tile boundary x=64
        # Compare pixels on either side of boundary
        left_strip = warped_frame[:, 62:64, :]
        right_strip = warped_frame[:, 64:66, :]

        # Should have smooth transition (difference should be small)
        # For checkerboard pattern, max difference at boundary should be < 0.5
        max_diff = np.max(np.abs(left_strip - right_strip))
        assert max_diff < 0.6, f"Tile boundary discontinuity detected: {max_diff}"

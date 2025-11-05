"""
Tests for FlowSRRefine node - flow upsampling and guided filtering.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_transfer_nodes import FlowSRRefine


class TestFlowSRRefine:
    """Test flow upsampling and edge-aware refinement."""

    def test_upscale_dimensions(self, sample_flow_fields, sample_still_image):
        """Test that flow is correctly upscaled to target dimensions."""
        node = FlowSRRefine()

        target_w, target_h = 256, 256

        upscaled = node.refine(sample_flow_fields, sample_still_image,
                              target_width=target_w, target_height=target_h,
                              guided_filter_radius=4, guided_filter_eps=1e-3)[0]

        assert upscaled.shape == (3, 256, 256, 2), \
            f"Expected (3, 256, 256, 2), got {upscaled.shape}"

    def test_flow_scaling(self):
        """Test that flow vectors are correctly scaled during upsampling."""
        node = FlowSRRefine()

        # Create low-res flow: 64x64 with constant (10, 5) flow
        low_res_flow = np.ones((1, 64, 64, 2), dtype=np.float32)
        low_res_flow[:, :, :, 0] = 10.0  # u
        low_res_flow[:, :, :, 1] = 5.0   # v

        # Create guide image 256x256
        guide = np.random.rand(1, 256, 256, 3).astype(np.float32)

        # Upscale to 256x256 (4x scaling)
        upscaled = node.refine(low_res_flow, guide,
                              target_width=256, target_height=256,
                              guided_filter_radius=4, guided_filter_eps=1e-3)[0]

        # Flow vectors should be scaled by 4x
        # Expected: u ≈ 40, v ≈ 20 (allowing for filtering)
        mean_u = np.mean(upscaled[:, :, :, 0])
        mean_v = np.mean(upscaled[:, :, :, 1])

        np.testing.assert_allclose(mean_u, 40.0, rtol=0.2,
                                   err_msg="Horizontal flow not scaled correctly")
        np.testing.assert_allclose(mean_v, 20.0, rtol=0.2,
                                   err_msg="Vertical flow not scaled correctly")

    def test_preserve_flow_direction(self):
        """Test that flow direction is preserved after upsampling."""
        node = FlowSRRefine()

        # Create diagonal flow (northeast direction)
        low_res_flow = np.ones((1, 32, 32, 2), dtype=np.float32)
        low_res_flow[:, :, :, 0] = 5.0   # right
        low_res_flow[:, :, :, 1] = -3.0  # up (negative Y)

        guide = np.random.rand(1, 128, 128, 3).astype(np.float32)

        upscaled = node.refine(low_res_flow, guide,
                              target_width=128, target_height=128,
                              guided_filter_radius=4, guided_filter_eps=1e-3)[0]

        # Check that direction is preserved (u positive, v negative)
        assert np.mean(upscaled[:, :, :, 0]) > 0, "Horizontal flow should be rightward"
        assert np.mean(upscaled[:, :, :, 1]) < 0, "Vertical flow should be upward"

    def test_batch_processing(self):
        """Test that batch of flows is processed correctly."""
        node = FlowSRRefine()

        # Create batch of 5 different flows
        batch_size = 5
        flows = []
        for i in range(batch_size):
            flow = np.ones((64, 64, 2), dtype=np.float32) * (i + 1)
            flows.append(flow)
        flow_batch = np.stack(flows, axis=0)

        guide = np.random.rand(1, 256, 256, 3).astype(np.float32)

        upscaled = node.refine(flow_batch, guide,
                              target_width=256, target_height=256,
                              guided_filter_radius=4, guided_filter_eps=1e-3)[0]

        assert upscaled.shape[0] == batch_size, \
            f"Expected batch size {batch_size}, got {upscaled.shape[0]}"

        # Each frame should have different magnitude
        magnitudes = [np.mean(np.abs(upscaled[i])) for i in range(batch_size)]

        # Magnitudes should be increasing (monotonic)
        for i in range(batch_size - 1):
            assert magnitudes[i] < magnitudes[i + 1], \
                "Flow magnitudes should increase across batch"

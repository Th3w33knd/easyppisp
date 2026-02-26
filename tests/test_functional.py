"""
Tests for the stateless functional API (easyppisp.functional).

Covers: identity, known-value, shape, batch, gradient flow.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from easyppisp.functional import (
    apply_exposure,
    apply_vignetting,
    apply_color_correction,
    apply_crf,
    apply_pipeline,
)
from easyppisp.validation import PPISPShapeError, PPISPPhysicsWarning


# ===========================================================================
# apply_exposure
# ===========================================================================


class TestApplyExposure:
    def test_known_value_double_brightness(self, img_hwc_ones):
        """+1 EV should exactly double the image values."""
        result = apply_exposure(img_hwc_ones, delta_t=1.0)
        assert torch.allclose(result, torch.tensor(2.0))

    def test_known_value_half_brightness(self, img_hwc_ones):
        """-1 EV should halve the image values."""
        result = apply_exposure(img_hwc_ones, delta_t=-1.0)
        assert torch.allclose(result, torch.tensor(0.5))

    def test_identity(self, img_hwc):
        """delta_t=0.0 must return a tensor equal to the input."""
        result = apply_exposure(img_hwc, delta_t=0.0)
        assert torch.allclose(result, img_hwc)

    def test_tensor_input(self, img_hwc):
        """Accepts a 0-D Tensor for delta_t."""
        dt = torch.tensor(1.0)
        result = apply_exposure(img_hwc, delta_t=dt)
        expected = apply_exposure(img_hwc, delta_t=1.0)
        assert torch.allclose(result, expected)

    def test_batch_shape_preserved(self, img_hwc_batch):
        """Output shape must match input for batched tensors."""
        result = apply_exposure(img_hwc_batch, delta_t=0.5)
        assert result.shape == img_hwc_batch.shape

    def test_bad_shape_raises(self):
        """1-D tensor must raise PPISPShapeError."""
        with pytest.raises(PPISPShapeError):
            apply_exposure(torch.ones(10), delta_t=0.0)

    def test_wrong_channels_raises(self):
        """RGBA (4-channel) must raise PPISPShapeError."""
        with pytest.raises(PPISPShapeError):
            apply_exposure(torch.ones(4, 4, 4), delta_t=0.0)

    def test_very_large_exposure_warns(self, img_hwc):
        """delta_t > 10 should emit PPISPPhysicsWarning."""
        with pytest.warns(PPISPPhysicsWarning):
            apply_exposure(img_hwc, delta_t=15.0)

    def test_preserves_dtype(self, img_hwc):
        """Output dtype must match input dtype."""
        assert apply_exposure(img_hwc, delta_t=1.0).dtype == img_hwc.dtype


# ===========================================================================
# apply_vignetting
# ===========================================================================


class TestApplyVignetting:
    def test_identity_at_zero_alpha(self, img_hwc, identity_alpha, identity_center):
        """Alpha=0, center=0 must return a tensor ≈ equal to input (clamp(1, 0, 1) = 1)."""
        result = apply_vignetting(img_hwc, identity_alpha, identity_center)
        assert torch.allclose(result, img_hwc)

    def test_output_shape_single(self, img_hwc):
        """Single image shape preserved."""
        alpha = torch.zeros(3, 3)
        center = torch.zeros(2)
        result = apply_vignetting(img_hwc, alpha, center)
        assert result.shape == img_hwc.shape

    def test_output_shape_batch(self, img_hwc_batch):
        """Batch shape preserved."""
        alpha = torch.zeros(3, 3)
        center = torch.zeros(2)
        result = apply_vignetting(img_hwc_batch, alpha, center)
        assert result.shape == img_hwc_batch.shape

    def test_negative_alpha_darkens_edges(self):
        """Negative α₁ should reduce brightness at the image edges."""
        img = torch.ones(32, 32, 3)
        alpha = torch.zeros(3, 3)
        alpha[:, 0] = -1.0   # strong r² falloff
        center = torch.zeros(2)
        result = apply_vignetting(img, alpha, center)
        # Corner pixels should be darker than center
        center_val = result[16, 16, 0].item()
        corner_val = result[0, 0, 0].item()
        assert corner_val < center_val

    def test_output_clamped_to_unit_range(self):
        """Falloff clamp ensures output never exceeds [0, 1] × input."""
        img = torch.ones(16, 16, 3) * 0.8
        alpha = torch.zeros(3, 3)
        alpha[:, 0] = 5.0   # huge positive alpha (would push > 1)
        center = torch.zeros(2)
        result = apply_vignetting(img, alpha, center)
        assert result.max().item() <= 0.8 + 1e-5

    def test_bad_shape_raises(self, identity_alpha, identity_center):
        with pytest.raises(PPISPShapeError):
            apply_vignetting(torch.ones(3, 4), identity_alpha, identity_center)


# ===========================================================================
# apply_color_correction
# ===========================================================================


class TestApplyColorCorrection:
    def test_identity_at_zero_offsets(self, img_hwc, identity_color_offsets):
        """Zero offsets must produce an output ≈ equal to input."""
        result = apply_color_correction(img_hwc, identity_color_offsets)
        # Homography is constructed from source chromaticities, so zero offsets
        # give H ≈ I.  Intensity normalization means result ≈ input.
        assert torch.allclose(result, img_hwc, atol=1e-4)

    def test_output_shape_preserved(self, img_hwc, identity_color_offsets):
        result = apply_color_correction(img_hwc, identity_color_offsets)
        assert result.shape == img_hwc.shape

    def test_batch_shape_preserved(self, img_hwc_batch, identity_color_offsets):
        result = apply_color_correction(img_hwc_batch, identity_color_offsets)
        assert result.shape == img_hwc_batch.shape

    def test_missing_key_raises(self, img_hwc):
        """Missing 'B' key must raise KeyError."""
        offsets = {"R": torch.zeros(2), "G": torch.zeros(2), "W": torch.zeros(2)}
        with pytest.raises(KeyError):
            apply_color_correction(img_hwc, offsets)

    def test_differentiable(self, img_hwc):
        """Gradients should flow through color correction."""
        offsets = {k: torch.zeros(2, requires_grad=True) for k in ("R", "G", "B", "W")}
        result = apply_color_correction(img_hwc, offsets)
        result.sum().backward()
        for k, v in offsets.items():
            assert v.grad is not None, f"No gradient for key '{k}'"


# ===========================================================================
# apply_crf
# ===========================================================================


class TestApplyCRF:
    def test_output_in_unit_range(self, img_hwc, identity_crf_raw):
        """CRF output must be in [0, 1]."""
        tau, eta, xi, gamma = identity_crf_raw
        result = apply_crf(img_hwc, tau, eta, xi, gamma)
        assert result.min().item() >= 0.0 - 1e-6
        assert result.max().item() <= 1.0 + 1e-6

    def test_output_shape_preserved(self, img_hwc, identity_crf_raw):
        tau, eta, xi, gamma = identity_crf_raw
        result = apply_crf(img_hwc, tau, eta, xi, gamma)
        assert result.shape == img_hwc.shape

    def test_batch_shape_preserved(self, img_hwc_batch, identity_crf_raw):
        tau, eta, xi, gamma = identity_crf_raw
        result = apply_crf(img_hwc_batch, tau, eta, xi, gamma)
        assert result.shape == img_hwc_batch.shape

    def test_monotone_channel_response(self, identity_crf_raw):
        """CRF must be monotonically non-decreasing for each channel."""
        tau, eta, xi, gamma = identity_crf_raw
        xs = torch.linspace(0.01, 0.99, 100).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
        ys = apply_crf(xs, tau, eta, xi, gamma)
        # Check per-channel monotonicity
        diffs = ys[1:] - ys[:-1]
        assert (diffs >= -1e-5).all(), "CRF is not monotonically increasing."

    def test_differentiable(self, img_hwc):
        """Gradients should flow through CRF."""
        img = img_hwc.detach().requires_grad_(True)
        tau = torch.zeros(3, requires_grad=True)
        eta = torch.zeros(3, requires_grad=True)
        xi = torch.zeros(3, requires_grad=True)
        gamma = torch.zeros(3, requires_grad=True)
        result = apply_crf(img, tau, eta, xi, gamma)
        result.sum().backward()
        assert img.grad is not None
        assert tau.grad is not None


# ===========================================================================
# apply_pipeline
# ===========================================================================


class TestApplyPipeline:
    def test_all_none_stages_identity(self, img_hwc):
        """No stages active (except default 0.0 exposure) should ≈ return input."""
        result = apply_pipeline(img_hwc, exposure_offset=0.0)
        assert torch.allclose(result, img_hwc)

    def test_exposure_applied(self, img_hwc_ones):
        """+1 EV through pipeline doubles brightness."""
        result = apply_pipeline(img_hwc_ones, exposure_offset=1.0)
        assert torch.allclose(result, torch.tensor(2.0))

    def test_shape_preserved(self, img_hwc):
        result = apply_pipeline(img_hwc)
        assert result.shape == img_hwc.shape

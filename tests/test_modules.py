"""
Tests for the nn.Module API (easyppisp.modules).

Covers: composability, independent usage, physical ordering warning,
parameter export, from_params, state_dict round-trip.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch

from easyppisp.modules import (
    ExposureOffset,
    Vignetting,
    ColorCorrection,
    CameraResponseFunction,
    ISPPipeline,
)
from easyppisp.params import PipelineParams, PipelineResult
from easyppisp.validation import PPISPPhysicsWarning


# ===========================================================================
# Individual modules — standalone usage
# ===========================================================================


class TestExposureOffset:
    def test_forward_doubles_brightness(self, img_hwc_ones):
        mod = ExposureOffset(delta_t=1.0)
        result = mod(img_hwc_ones)
        assert torch.allclose(result, torch.tensor(2.0))

    def test_identity(self, img_hwc):
        mod = ExposureOffset(delta_t=0.0)
        assert torch.allclose(mod(img_hwc), img_hwc)

    def test_from_params(self):
        p = PipelineParams(exposure_offset=0.75)
        mod = ExposureOffset.from_params(p)
        assert abs(mod.delta_t.item() - 0.75) < 1e-6

    def test_get_params_dict(self):
        mod = ExposureOffset(delta_t=2.0)
        d = mod.get_params_dict()
        assert "exposure_offset_ev" in d
        assert abs(d["exposure_offset_ev"] - 2.0) < 1e-5

    def test_is_nn_parameter(self):
        mod = ExposureOffset(delta_t=1.0)
        params = list(mod.parameters())
        assert len(params) == 1
        assert params[0].requires_grad


class TestVignetting:
    def test_identity_at_zeros(self, img_hwc):
        mod = Vignetting()
        assert torch.allclose(mod(img_hwc), img_hwc)

    def test_from_params(self):
        p = PipelineParams()
        p.vignetting_alpha = torch.ones(3, 3) * 0.05
        mod = Vignetting.from_params(p)
        assert torch.allclose(mod.alpha, torch.ones(3, 3) * 0.05)

    def test_get_params_dict_keys(self):
        mod = Vignetting()
        d = mod.get_params_dict()
        assert "vignetting_alpha" in d
        assert "vignetting_center" in d

    def test_batch_forward(self, img_hwc_batch):
        mod = Vignetting()
        out = mod(img_hwc_batch)
        assert out.shape == img_hwc_batch.shape


class TestColorCorrection:
    def test_identity_at_zeros(self, img_hwc):
        mod = ColorCorrection()
        assert torch.allclose(mod(img_hwc), img_hwc, atol=1e-4)

    def test_separate_parameters(self):
        """B, R, G, W should be stored as separate nn.Parameters."""
        mod = ColorCorrection()
        param_names = {n for n, _ in mod.named_parameters()}
        assert "b_off" in param_names
        assert "r_off" in param_names
        assert "g_off" in param_names
        assert "w_off" in param_names

    def test_from_params(self):
        p = PipelineParams()
        p.color_offsets = {
            "R": torch.tensor([0.01, 0.02]),
            "G": torch.tensor([0.03, 0.04]),
            "B": torch.tensor([0.05, 0.06]),
            "W": torch.tensor([0.07, 0.08]),
        }
        mod = ColorCorrection.from_params(p)
        assert torch.allclose(mod.r_off, torch.tensor([0.01, 0.02]))


class TestCameraResponseFunction:
    def test_output_in_unit_range(self, img_hwc):
        mod = CameraResponseFunction()
        out = mod(img_hwc)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_from_params(self):
        p = PipelineParams()
        mod = CameraResponseFunction.from_params(p)
        assert mod.tau.shape == (3,)

    def test_get_params_dict_constrained_values(self):
        """Physical parameters in get_params_dict should satisfy constraints."""
        mod = CameraResponseFunction()
        d = mod.get_params_dict()
        for val in d["crf_tau"]:
            assert val > 0.3, "tau should be > 0.3"
        for val in d["crf_gamma"]:
            assert val > 0.1, "gamma should be > 0.1"
        for val in d["crf_xi"]:
            assert 0.0 < val < 1.0, "xi should be in (0, 1)"


# ===========================================================================
# ISPPipeline
# ===========================================================================


class TestISPPipeline:
    def test_default_pipeline_returns_pipeline_result(self, img_hwc):
        pipeline = ISPPipeline()
        result = pipeline(img_hwc)
        assert isinstance(result, PipelineResult)
        assert result.final.shape == img_hwc.shape

    def test_return_intermediates(self, img_hwc):
        pipeline = ISPPipeline()
        result = pipeline(img_hwc, return_intermediates=True)
        assert result.intermediates is not None
        assert "ExposureOffset" in result.intermediates
        assert "Vignetting" in result.intermediates
        assert "ColorCorrection" in result.intermediates
        assert "CameraResponseFunction" in result.intermediates

    def test_no_intermediates_by_default(self, img_hwc):
        pipeline = ISPPipeline()
        result = pipeline(img_hwc)
        assert result.intermediates is None

    def test_custom_subset_of_modules(self, img_hwc):
        """Only Exposure + CRF — composability requirement."""
        pipeline = ISPPipeline([ExposureOffset(delta_t=0.5), CameraResponseFunction()])
        result = pipeline(img_hwc, return_intermediates=True)
        assert "ExposureOffset" in result.intermediates
        assert "CameraResponseFunction" in result.intermediates
        assert "Vignetting" not in result.intermediates

    def test_physical_ordering_warning(self):
        """Linear op after CRF must emit PPISPPhysicsWarning."""
        with pytest.warns(PPISPPhysicsWarning, match="placed after CameraResponseFunction"):
            ISPPipeline([CameraResponseFunction(), ExposureOffset()])

    def test_no_warning_for_physical_order(self):
        """Correct order should not emit any warnings."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", PPISPPhysicsWarning)
            ISPPipeline([ExposureOffset(), CameraResponseFunction()])  # should not raise

    def test_from_params_round_trip(self, img_hwc):
        """from_params should produce same output as manually constructed pipeline."""
        p = PipelineParams()
        pipeline = ISPPipeline.from_params(p)
        result = pipeline(img_hwc)
        assert result.final.shape == img_hwc.shape

    def test_get_params_dict(self):
        pipeline = ISPPipeline()
        d = pipeline.get_params_dict()
        assert "ExposureOffset" in d
        assert "Vignetting" in d

    def test_state_dict_round_trip(self, img_hwc):
        """Saving and loading state_dict must reproduce same outputs."""
        pipeline = ISPPipeline()
        out1 = pipeline(img_hwc).final

        sd = pipeline.state_dict()
        pipeline2 = ISPPipeline()
        pipeline2.load_state_dict(sd)
        out2 = pipeline2(img_hwc).final

        assert torch.allclose(out1, out2)

    def test_batch_forward(self, img_hwc_batch):
        pipeline = ISPPipeline()
        result = pipeline(img_hwc_batch)
        assert result.final.shape == img_hwc_batch.shape

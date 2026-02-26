"""
Integration tests for the full ISP pipeline and task-level workflows.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch

import easyppisp
from easyppisp.modules import ISPPipeline, ExposureOffset, CameraResponseFunction
from easyppisp.tasks import CameraSimulator, PhysicalAugmentation, CameraMatchPair
from easyppisp.presets import FilmPreset
from easyppisp.params import PipelineParams


class TestPipelineEndToEnd:
    def test_default_pipeline_output_range(self, img_hwc):
        """Default ISPPipeline output must be in [0, 1]."""
        pipeline = ISPPipeline()
        out = pipeline(img_hwc).final
        assert out.min() >= -1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_pipeline_differentiable(self, img_hwc):
        """Loss gradients must flow through a complete pipeline."""
        pipeline = ISPPipeline()
        img = img_hwc.detach().requires_grad_(True)
        out = pipeline(img).final
        loss = out.sum()
        loss.backward()
        assert img.grad is not None
        # At least some pipeline parameters should have gradients
        param_grads = [p.grad for p in pipeline.parameters() if p.grad is not None]
        assert len(param_grads) > 0

    def test_exposure_change_propagates(self, img_hwc):
        """Changing ExposureOffset.delta_t must change the output."""
        p1 = ISPPipeline([ExposureOffset(delta_t=0.0), CameraResponseFunction()])
        p2 = ISPPipeline([ExposureOffset(delta_t=2.0), CameraResponseFunction()])
        out1 = p1(img_hwc).final
        out2 = p2(img_hwc).final
        assert not torch.allclose(out1, out2)


class TestEasyppisApply:
    def test_exposure_only_quickstart(self, img_hwc_ones):
        """Top-level apply() with exposure=1.0 must double brightness."""
        result = easyppisp.apply(img_hwc_ones, exposure=1.0)
        assert torch.allclose(result, torch.tensor(2.0))

    def test_apply_with_preset(self, img_hwc):
        """apply() with a named preset must return an image in [0, 1]."""
        result = easyppisp.apply(img_hwc, preset="kodak_portra_400")
        assert result.shape == img_hwc.shape
        assert result.min() >= -1e-5
        assert result.max() <= 1.0 + 1e-5


class TestCameraSimulator:
    def test_default_simulator_output_shape(self, img_hwc):
        cam = CameraSimulator()
        out = cam(img_hwc)
        assert out.shape == img_hwc.shape

    def test_preset_simulator(self, img_hwc):
        cam = CameraSimulator("kodak_portra_400")
        out = cam(img_hwc)
        assert out.shape == img_hwc.shape
        assert out.min() >= -1e-5

    def test_from_preset_classmethod(self, img_hwc):
        cam = CameraSimulator.from_preset("fuji_velvia_50")
        out = cam(img_hwc)
        assert out.shape == img_hwc.shape

    def test_set_exposure_changes_output(self, img_hwc):
        cam = CameraSimulator()
        out_default = cam(img_hwc)
        cam.set_exposure(1.0)
        out_bright = cam(img_hwc)
        # Brighter exposure should produce larger values on average
        assert out_bright.mean() > out_default.mean()

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            CameraSimulator("nonexistent_preset_xyz")


class TestPhysicalAugmentation:
    def test_output_shape(self, img_hwc):
        aug = PhysicalAugmentation()
        out = aug(img_hwc)
        assert out.shape == img_hwc.shape

    def test_output_clamped(self, img_hwc):
        aug = PhysicalAugmentation(exposure_range=(-5.0, 5.0))
        out = aug(img_hwc)
        assert out.min() >= -1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_different_calls_produce_different_results(self, img_hwc):
        """With a wide exposure range, repeated calls should differ."""
        aug = PhysicalAugmentation(exposure_range=(-3.0, 3.0))
        results = [aug(img_hwc) for _ in range(10)]
        # At least some pairs should differ
        any_differ = any(not torch.allclose(results[0], r) for r in results[1:])
        assert any_differ

    def test_batch_compatible(self, img_hwc_batch):
        aug = PhysicalAugmentation()
        out = aug(img_hwc_batch)
        assert out.shape == img_hwc_batch.shape


class TestFilmPresets:
    def test_list_presets_non_empty(self):
        names = FilmPreset.list_presets()
        assert len(names) > 0
        assert "default" in names
        assert "kodak_portra_400" in names

    def test_load_default_is_identity(self, img_hwc):
        """Default preset with identity params should produce output in [0,1]."""
        pipeline = FilmPreset.load("default")
        out = pipeline(img_hwc).final
        assert out.shape == img_hwc.shape

    def test_load_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            FilmPreset.load("this_does_not_exist")

    def test_save_and_load_custom_preset(self, img_hwc, tmp_path):
        p = PipelineParams(exposure_offset=1.0)
        path = tmp_path / "custom.json"
        FilmPreset.save_params("custom", p, path)
        pipeline = FilmPreset.load_from_file(path)
        out = pipeline(img_hwc)
        assert out.final.shape == img_hwc.shape


class TestCameraMatchPair:
    def test_fit_and_transform(self, img_hwc):
        """fit() + transform() should return an image of the correct shape."""
        src = [img_hwc]
        tgt = [img_hwc * 0.7]   # target is slightly darker
        matcher = CameraMatchPair()
        matcher.fit(src, tgt, num_steps=20, verbose=False)
        out = matcher.transform(img_hwc)
        assert out.shape == img_hwc.shape

    def test_transform_before_fit_raises(self, img_hwc):
        matcher = CameraMatchPair()
        with pytest.raises(RuntimeError, match="fit"):
            matcher.transform(img_hwc)

    def test_save_params(self, img_hwc, tmp_path):
        src = [img_hwc]
        tgt = [img_hwc]
        matcher = CameraMatchPair()
        matcher.fit(src, tgt, num_steps=5, verbose=False)
        path = tmp_path / "match.json"
        matcher.save_params(str(path))
        assert path.exists()


class TestIdentityInvariant:
    """Verify that default parameters produce an identity (no-op) transform."""

    def test_default_params_pipeline_is_identity(self, img_hwc):
        """ISPPipeline.from_params(PipelineParams()) must return input unchanged."""
        from easyppisp.params import PipelineParams
        from easyppisp.modules import ISPPipeline
        pipeline = ISPPipeline.from_params(PipelineParams())
        out = pipeline(img_hwc).final
        assert torch.allclose(out, img_hwc, atol=1e-4), (
            f"Default pipeline is not identity: max diff = {(out - img_hwc).abs().max().item():.6f}"
        )

    def test_identity_preset_is_identity(self, img_hwc):
        """'identity' FilmPreset must also produce output ≈ input."""
        from easyppisp.presets import FilmPreset
        pipeline = FilmPreset.load("identity")
        out = pipeline(img_hwc).final
        assert torch.allclose(out, img_hwc, atol=1e-4)

    def test_functional_apply_pipeline_no_stages_is_identity(self, img_hwc):
        """apply_pipeline with no active stages must return input unchanged."""
        from easyppisp.functional import apply_pipeline
        out = apply_pipeline(img_hwc, exposure_offset=0.0)
        assert torch.allclose(out, img_hwc)

    def test_apply_top_level_zero_exposure_is_identity(self, img_hwc):
        """easyppisp.apply(img) with no args must return input unchanged."""
        import easyppisp
        out = easyppisp.apply(img_hwc)
        assert torch.allclose(out, img_hwc)


class TestSerializationInterop:
    """Verify that save/load round-trips preserve parameter values."""

    def test_camera_match_save_loadable_by_pipeline_params(self, img_hwc, tmp_path):
        """CameraMatchPair.save_params() output must be loadable by PipelineParams.load()."""
        from easyppisp.tasks import CameraMatchPair
        from easyppisp.params import PipelineParams

        matcher = CameraMatchPair()
        matcher.fit([img_hwc], [img_hwc * 0.8], num_steps=5, verbose=False)
        path = str(tmp_path / "match.json")
        matcher.save_params(path)

        # Must load without error
        p = PipelineParams.load(path)
        assert isinstance(p, PipelineParams)
        # Exposure key must be present and numeric
        assert isinstance(p.exposure_offset, float)

    def test_pipeline_params_crf_round_trip(self, tmp_path):
        """PipelineParams save → load must recover CRF raw values exactly."""
        import torch
        from easyppisp.params import PipelineParams
        p = PipelineParams()
        # Perturb CRF to non-default values
        p.crf_tau = torch.tensor([0.1, 0.2, 0.3])
        path = str(tmp_path / "crf_test.json")
        p.save(path)
        p2 = PipelineParams.load(path)
        assert torch.allclose(p2.crf_tau, p.crf_tau, atol=1e-6)

    def test_apply_with_kwargs_no_silent_ignore(self, img_hwc):
        """Passing color_offsets to apply() must actually apply them (not silently ignore)."""
        import torch, easyppisp
        zero_offsets = {k: torch.zeros(2) for k in ("R", "G", "B", "W")}
        # With zero offsets, output should match plain exposure-only call
        out_with = easyppisp.apply(img_hwc, exposure=0.5, color_offsets=zero_offsets)
        out_without = easyppisp.apply(img_hwc, exposure=0.5)
        assert torch.allclose(out_with, out_without, atol=1e-4)

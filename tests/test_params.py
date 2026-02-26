"""
Tests for PipelineParams and PipelineResult serialization.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from easyppisp.params import PipelineParams, PipelineResult


class TestPipelineParams:
    def test_default_construction(self):
        p = PipelineParams()
        assert p.exposure_offset == 0.0
        assert p.vignetting_alpha.shape == (3, 3)
        assert p.vignetting_center.shape == (2,)
        assert set(p.color_offsets.keys()) == {"R", "G", "B", "W"}
        for v in p.color_offsets.values():
            assert v.shape == (2,)

    def test_to_dict_roundtrip(self):
        """to_dict → from_dict must recover equivalent values."""
        p = PipelineParams(
            exposure_offset=1.5,
            vignetting_alpha=torch.tensor([[0.1, 0.2, 0.3]] * 3),
            color_offsets={
                "R": torch.tensor([0.01, 0.02]),
                "G": torch.tensor([0.03, 0.04]),
                "B": torch.tensor([0.05, 0.06]),
                "W": torch.tensor([0.07, 0.08]),
            },
        )
        d = p.to_dict()
        p2 = PipelineParams.from_dict(d)

        assert abs(p2.exposure_offset - p.exposure_offset) < 1e-6
        assert torch.allclose(p2.vignetting_alpha, p.vignetting_alpha, atol=1e-6)
        assert torch.allclose(p2.color_offsets["R"], p.color_offsets["R"], atol=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        """Save to JSON → load from JSON must preserve all values."""
        p = PipelineParams(exposure_offset=0.75)
        path = tmp_path / "params.json"
        p.save(str(path))
        p2 = PipelineParams.load(str(path))
        assert abs(p2.exposure_offset - 0.75) < 1e-6

    def test_saved_file_is_valid_json(self, tmp_path):
        p = PipelineParams()
        path = tmp_path / "params.json"
        p.save(str(path))
        with open(path) as f:
            data = json.load(f)
        assert "exposure_offset" in data
        assert "color_offsets" in data

    def test_color_offsets_are_tensors(self):
        """color_offsets values must be Tensors (not lists) for gradient flow."""
        p = PipelineParams()
        for k, v in p.color_offsets.items():
            assert isinstance(v, torch.Tensor), f"color_offsets['{k}'] must be a Tensor"


class TestPipelineResult:
    def test_construction(self):
        final = torch.rand(4, 4, 3)
        result = PipelineResult(final=final)
        assert result.final is final
        assert result.intermediates is None
        assert result.params_used is None

    def test_with_intermediates(self):
        final = torch.rand(4, 4, 3)
        intermediates = {"Exposure": torch.rand(4, 4, 3)}
        result = PipelineResult(final=final, intermediates=intermediates)
        assert "Exposure" in result.intermediates

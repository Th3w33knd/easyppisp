"""
Gradient flow tests using torch.autograd.gradcheck.

All tests use float64 tensors (gradcheck requires double precision for accurate
finite-difference approximation).  Small images (4×4) are used for speed.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from easyppisp.functional import (
    apply_exposure,
    apply_vignetting,
    apply_color_correction,
    apply_crf,
)


def _rand64(shape, low=0.1, high=0.9) -> torch.Tensor:
    """float64 tensor with values safely away from clamp boundaries."""
    return (torch.rand(shape, dtype=torch.float64) * (high - low) + low).requires_grad_(True)


def _zeros64(shape) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float64, requires_grad=True)


class TestGradcheck:
    def test_exposure_gradcheck(self):
        img = _rand64((3, 3, 3))
        dt = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(apply_exposure, (img, dt), eps=1e-4, atol=1e-3, fast_mode=True)

    def test_vignetting_gradcheck(self):
        img = _rand64((4, 4, 3))
        alpha = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        center = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            apply_vignetting, (img, alpha, center), eps=1e-4, atol=1e-3, fast_mode=True
        )

    def test_vignetting_nonzero_gradcheck(self):
        """Gradcheck with non-trivial alpha (partial falloff at test resolution)."""
        img = _rand64((4, 4, 3))
        # Small alpha to avoid clamp saturation
        alpha = torch.full((3, 3), -0.05, dtype=torch.float64, requires_grad=True)
        center = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            apply_vignetting, (img, alpha, center), eps=1e-4, atol=1e-3, fast_mode=True
        )

    def test_crf_gradcheck(self):
        """CRF uses clamp internally; restrict input to interior to avoid non-differentiable edges."""
        img = _rand64((3, 3, 3), low=0.2, high=0.8)
        tau_r   = _zeros64((3,))
        eta_r   = _zeros64((3,))
        xi_r    = _zeros64((3,))
        gamma_r = _zeros64((3,))
        assert gradcheck(
            apply_crf, (img, tau_r, eta_r, xi_r, gamma_r),
            eps=1e-4, atol=1e-3, fast_mode=True
        )

    def test_color_correction_gradcheck(self):
        """Wrap apply_color_correction to accept flat tensors (gradcheck requirement)."""
        img = _rand64((3, 3, 3))
        b = _zeros64((2,))
        r = _zeros64((2,))
        g = _zeros64((2,))
        w = _zeros64((2,))

        def _wrapper(img_t, b_t, r_t, g_t, w_t):
            return apply_color_correction(
                img_t, {"B": b_t, "R": r_t, "G": g_t, "W": w_t}
            )

        assert gradcheck(
            _wrapper, (img, b, r, g, w), eps=1e-5, atol=1e-3, fast_mode=True
        )

    def test_exposure_gradient_value(self):
        """∂(apply_exposure)/∂(delta_t) = image * ln(2) * 2^delta_t."""
        img = torch.ones(2, 2, 3, dtype=torch.float64)
        dt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        out = apply_exposure(img, dt)
        out.sum().backward()
        expected_grad = img.sum().item() * torch.log(torch.tensor(2.0)).item()
        assert abs(dt.grad.item() - expected_grad) < 1e-5

"""
Shared pytest fixtures for the easyppisp test suite.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def img_hwc() -> Tensor:
    """Small single-image (H=8, W=8, C=3) float32 in [0.1, 0.9]."""
    return torch.rand(8, 8, 3, dtype=torch.float32) * 0.8 + 0.1


@pytest.fixture
def img_hwc_batch() -> Tensor:
    """Small batched image (B=2, H=8, W=8, C=3) float32 in [0.1, 0.9]."""
    return torch.rand(2, 8, 8, 3, dtype=torch.float32) * 0.8 + 0.1


@pytest.fixture
def img_hwc_ones() -> Tensor:
    """All-ones single image (H=4, W=4, C=3)."""
    return torch.ones(4, 4, 3, dtype=torch.float32)


@pytest.fixture
def img_hwc_f64() -> Tensor:
    """float64 single image for gradcheck (avoids float32 precision issues)."""
    return (torch.rand(4, 4, 3, dtype=torch.float64) * 0.8 + 0.1).requires_grad_(True)


# ---------------------------------------------------------------------------
# Parameter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_alpha() -> Tensor:
    """Vignetting alpha = zeros → identity."""
    return torch.zeros(3, 3, dtype=torch.float32)


@pytest.fixture
def identity_center() -> Tensor:
    """Vignetting center = zeros → image center."""
    return torch.zeros(2, dtype=torch.float32)


@pytest.fixture
def identity_color_offsets() -> dict:
    """Color offsets = zeros → identity homography."""
    return {k: torch.zeros(2, dtype=torch.float32) for k in ("R", "G", "B", "W")}


from easyppisp.params import (
    _CRF_TAU_IDENTITY,
    _CRF_ETA_IDENTITY,
    _CRF_XI_IDENTITY,
    _CRF_GAMMA_IDENTITY,
)


@pytest.fixture
def identity_crf_raw() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """CRF raw params that produce exactly tau=eta=1, xi=0.5, gamma=1 (identity)."""
    tau = torch.full((3,), _CRF_TAU_IDENTITY, dtype=torch.float32)
    eta = torch.full((3,), _CRF_ETA_IDENTITY, dtype=torch.float32)
    xi = torch.full((3,), _CRF_XI_IDENTITY, dtype=torch.float32)
    gamma = torch.full((3,), _CRF_GAMMA_IDENTITY, dtype=torch.float32)
    return tau, eta, xi, gamma


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=["cpu"])
def device(request) -> str:
    """Device to run tests on. CUDA added only when available."""
    return request.param


def pytest_configure(config):
    """Register CUDA marker."""
    config.addinivalue_line("markers", "cuda: requires a CUDA GPU")

"""
Regularization losses for ISP optimization (Eq. 18–22 of the PPISP paper).

These losses help resolve ambiguities (like Exposure ↔ SH coefficients) and
encourage physically plausible parameter states.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._internal.color_homography import _COLOR_PINV_BLOCK_DIAG


def exposure_mean_loss(exposure_params: Tensor, target: float = 0.0) -> Tensor:
    """Encourage average exposure offset to be near zero (Eq. 18).

    Resolves the ambiguity between per-frame exposure and spherical harmonics.
    Uses Smooth L1 loss for robustness.
    """
    mean_exp = exposure_params.mean()
    return F.smooth_l1_loss(
        mean_exp, torch.tensor(target, device=mean_exp.device, dtype=mean_exp.dtype), beta=0.1
    )


def vignetting_center_loss(center: Tensor) -> Tensor:
    """Encourage the vignetting optical center to be near the image center (Eq. 19).

    Args:
        center: Optical center offset ``(..., 2)`` in normalized coordinates.
    """
    return (center**2).sum(dim=-1).mean()


def vignetting_non_pos_loss(alpha: Tensor) -> Tensor:
    """Penalize positive vignetting coefficients (Eq. 20).

    Vignetting should only decrease intensity radially (α ≤ 0).
    """
    return F.relu(alpha).mean()


def vignetting_channel_var_loss(alpha: Tensor) -> Tensor:
    """Encourage similar vignetting across RGB channels (Eq. 21).

    Args:
        alpha: Vignetting coefficients ``(..., 3, 3)`` [camera, channel, poly_term].
    """
    if alpha.ndim < 2:
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)
    return alpha.var(dim=-2, unbiased=False).mean()


def color_mean_loss(latent_offsets: Tensor) -> Tensor:
    """Encourage the average color correction to be near identity (Eq. 22).

    Maps latent offsets to real chromaticity space via ZCA preconditioning
    before applying the penalty.
    """
    # Map to real chromaticity space: (N, 8) @ (8, 8) -> (N, 8)
    real_offsets = latent_offsets @ _COLOR_PINV_BLOCK_DIAG.to(
        device=latent_offsets.device, dtype=latent_offsets.dtype
    )
    mean_color = real_offsets.mean(dim=0)
    return F.smooth_l1_loss(
        mean_color, torch.zeros_like(mean_color), beta=0.005
    )


def crf_channel_var_loss(
    tau_raw: Tensor, eta_raw: Tensor, xi_raw: Tensor, gamma_raw: Tensor
) -> Tensor:
    """Encourage similar CRF parameters across RGB channels.

    Ensures the sensor response doesn't introduce extreme per-channel
    non-linearities unless the data strongly supports it.
    """
    loss = (
        tau_raw.var(dim=-1, unbiased=False).mean()
        + eta_raw.var(dim=-1, unbiased=False).mean()
        + xi_raw.var(dim=-1, unbiased=False).mean()
        + gamma_raw.var(dim=-1, unbiased=False).mean()
    )
    return loss / 4.0

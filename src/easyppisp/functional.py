"""
Functional API for easyppisp — stateless, differentiable pure functions.

Each function corresponds directly to one stage of the PPISP physical camera model:

  Stage 1 — Exposure:        apply_exposure()         Eq. (3)
  Stage 2 — Vignetting:      apply_vignetting()       Eq. (4)–(5)
  Stage 3 — Color Correction: apply_color_correction() Eq. (6)–(12)
  Stage 4 — CRF / Tone map:  apply_crf()             Eq. (13)–(16)
  Full pipeline:              apply_pipeline()         (composes all four)

All functions:
  - Accept single images (H, W, 3) **and** batches (B, H, W, 3)
  - Are fully differentiable (safe for use in autograd / optimization loops)
  - Default to the HWC (channels-last) convention matching the paper and PPISP library
  - Delegate math to the PPISP CUDA backend when available, with a pure-PyTorch fallback

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .validation import (
    PPISPShapeError,
    PPISPPhysicsWarning,
    check_image_shape,
    check_exposure_range,
)
from ._internal.color_homography import build_homography, apply_homography

if TYPE_CHECKING:
    pass

logger = logging.getLogger("easyppisp")

# This module uses pure-PyTorch implementations for all pipeline stages.
# The PPISP CUDA kernel (ppisp) can be imported by callers directly for
# production training; easyppisp wraps the math at the Python level.


# ---------------------------------------------------------------------------
# Stage 1: Exposure
# ---------------------------------------------------------------------------


def apply_exposure(image: Tensor, delta_t: float | Tensor) -> Tensor:
    """Apply an exposure offset to a linear radiance image.

    Implements ``I_exp = L · 2^(Δt)`` (Eq. 3).

    Args:
        image: Linear radiance image ``(H, W, 3)`` or ``(B, H, W, 3)``.
        delta_t: Exposure offset in **EV (exposure value / stops)**.
            ``+1.0`` doubles the brightness; ``-1.0`` halves it.
            Pass a scalar ``float`` or a 0-D or broadcastable ``Tensor``.

    Returns:
        Exposure-adjusted image of the same shape as *image*.

    Example:
        >>> img = torch.ones(4, 4, 3)
        >>> result = apply_exposure(img, delta_t=1.0)
        >>> result.mean().item()  # 2.0 (one stop brighter)
        2.0
    """
    check_image_shape(image)
    if isinstance(delta_t, (int, float)):
        check_exposure_range(delta_t)
        delta_t = torch.tensor(float(delta_t), device=image.device, dtype=image.dtype)
    else:
        check_exposure_range(delta_t)
        delta_t = delta_t.to(device=image.device, dtype=image.dtype)

    return image * torch.pow(torch.tensor(2.0, device=image.device, dtype=image.dtype), delta_t)


# ---------------------------------------------------------------------------
# Stage 2: Vignetting
# ---------------------------------------------------------------------------


def apply_vignetting(
    image: Tensor,
    alpha: Tensor,
    center: Tensor,
    pixel_coords: Tensor | None = None,
) -> Tensor:
    """Apply per-channel radial vignetting (lens falloff).

    Implements the polynomial falloff model (Eq. 4–5):
    ``I_vig = I · clamp(1 + α₁r² + α₂r⁴ + α₃r⁶, 0, 1)``

    where ``r = ‖(u − μ) / max(H, W)‖₂`` is the normalized distance from
    the optical center ``μ``.  Normalization by ``max(H, W)`` matches the
    PPISP CUDA kernel's coordinate convention.

    Args:
        image: Input image ``(H, W, 3)`` or ``(B, H, W, 3)``.
        alpha: Per-channel polynomial coefficients ``(3, 3)``.
            Shape is ``[channel (RGB), polynomial term (α₁, α₂, α₃)]``.
            Identity = all zeros.
        center: Optical center offset ``(2,)`` relative to the image center,
            in the same normalized coordinate space as r.
            Identity = ``[0.0, 0.0]``.
        pixel_coords: Optional pre-computed pixel coordinate grid ``(..., H, W, 2)``.
            If None, a default grid is generated from the image resolution.

    Returns:
        Vignetting-corrected image of the same shape as *image*.
    """
    check_image_shape(image)

    ndim = image.ndim
    H, W = image.shape[-3], image.shape[-2]   # works for both 3D and 4D
    max_dim = float(max(H, W))
    device, dtype = image.device, image.dtype

    if pixel_coords is None:
        # Generate default integer pixel grid, anchored at pixel centers.
        # Normalize to [-0.5, 0.5] × [-0.5 * (H/W), 0.5 * (H/W)] approximately.
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each
        uv = torch.stack([grid_x, grid_y], dim=-1)                # (H, W, 2)
    else:
        uv = pixel_coords.to(device=device, dtype=dtype)

    # Normalize: shift to image center, divide by max dimension (PPISP convention)
    uv_norm = (uv - torch.tensor([W / 2.0, H / 2.0], device=device, dtype=dtype)) / max_dim

    # Optical center in the same normalized space
    center_n = center.to(device=device, dtype=dtype)   # (2,)

    # Reshape center for broadcasting: (1, 1, 2) for 3D or (1, 1, 1, 2) for 4D
    extra_dims = ndim - 1   # number of spatial + batch dims before the channel dim
    center_v = center_n.view(*([1] * extra_dims), 2)

    # Expand uv_norm for batch dimension if needed
    if ndim == 4:
        uv_norm = uv_norm.unsqueeze(0)   # (1, H, W, 2) — broadcasts over B

    delta = uv_norm - center_v           # (..., H, W, 2)
    r2 = (delta * delta).sum(dim=-1, keepdim=True)   # (..., H, W, 1)

    # Build per-channel polynomial falloff: 1 + α₁r² + α₂r⁴ + α₃r⁶
    # alpha has shape (3, 3): [RGB channel, polynomial term]
    alpha_dev = alpha.to(device=device, dtype=dtype)
    falloff = torch.ones_like(image)   # (..., H, W, 3)
    r2_pow = r2.expand_as(image)       # (..., H, W, 3) — same r for all channels initially
    r2_scalar = r2.expand_as(image)    # kept for successive powers

    for term_idx in range(3):
        # alpha_dev[:, term_idx] has shape (3,) = [R_coeff, G_coeff, B_coeff]
        coeff = alpha_dev[:, term_idx].view(*([1] * extra_dims), 3)   # (..., 1, 1, 3)
        if term_idx == 0:
            r_power = r2_scalar                                         # r²
        else:
            r_power = r_power * r2_scalar                               # r⁴, r⁶, ...
        falloff = falloff + coeff * r_power

    return image * falloff.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 3: Color Correction
# ---------------------------------------------------------------------------


def apply_color_correction(
    image: Tensor,
    color_offsets: dict[str, Tensor],
) -> Tensor:
    """Apply chromaticity homography-based color correction (white balance).

    Implements the full pipeline of Eq. (6)–(12): latent chromaticity offsets
    are mapped through the ZCA preconditioning matrix, then used to build a
    3×3 homography that is applied in RGI space.

    The internal ordering is always **B → R → G → W** (matching the ZCA
    block-diagonal layout).  The *color_offsets* dict can use any key order;
    this function extracts them in the correct sequence.

    Args:
        image: Input image ``(H, W, 3)`` or ``(B, H, W, 3)`` in linear space.
        color_offsets: Dict with keys ``'R'``, ``'G'``, ``'B'``, ``'W'``, each
            mapping to a ``(2,)`` tensor of ``[Δr, Δg]`` chromaticity offsets
            in latent ZCA space. Identity = all zeros.

    Returns:
        Color-corrected image of the same shape as *image*.

    Raises:
        KeyError: If any of 'R', 'G', 'B', 'W' is missing from *color_offsets*.
    """
    check_image_shape(image)

    required = ("R", "G", "B", "W")
    for key in required:
        if key not in color_offsets:
            raise KeyError(
                f"color_offsets is missing key '{key}'. "
                f"Expected keys: {required}. Got: {list(color_offsets.keys())}"
            )

    # Enforce B, R, G, W ordering (matches _COLOR_PINV_BLOCK_DIAG block layout)
    latent_offsets = torch.cat([
        color_offsets["B"].to(image.device, image.dtype),
        color_offsets["R"].to(image.device, image.dtype),
        color_offsets["G"].to(image.device, image.dtype),
        color_offsets["W"].to(image.device, image.dtype),
    ])  # (8,)

    H_mat = build_homography(latent_offsets)
    return apply_homography(image, H_mat)


# ---------------------------------------------------------------------------
# Stage 4: Camera Response Function (CRF)
# ---------------------------------------------------------------------------


def apply_crf(
    image: Tensor,
    tau_raw: Tensor,
    eta_raw: Tensor,
    xi_raw: Tensor,
    gamma_raw: Tensor,
) -> Tensor:
    """Apply the piecewise-power camera response function with gamma correction.

    Implements Eq. (13)–(16).  The *_raw* parameters are physically constrained
    inside this function via ``softplus`` / ``sigmoid`` to guarantee:
      - Monotonicity of the S-curve (tau > 0, eta > 0, 0 < xi < 1)
      - Positive gamma

    This means the raw parameters passed to the optimizer are unconstrained
    (gradient-friendly), while the forward computation always stays physical.

    Constraints (matching PPISP source):
      - ``tau   = 0.3 + softplus(tau_raw)``    → tau ∈ (0.3, ∞)
      - ``eta   = 0.3 + softplus(eta_raw)``    → eta ∈ (0.3, ∞)
      - ``xi    = sigmoid(xi_raw)``            → xi  ∈ (0,   1)
      - ``gamma = 0.1 + softplus(gamma_raw)``  → gamma ∈ (0.1, ∞)

    Args:
        image: Input image ``(H, W, 3)`` or ``(B, H, W, 3)`` in [0, 1].
        tau_raw: (3,) raw shadow power per channel.
        eta_raw: (3,) raw highlight power per channel.
        xi_raw:  (3,) raw inflection point per channel.
        gamma_raw: (3,) raw gamma per channel.

    Returns:
        Tone-mapped image of the same shape as *image*, values in [0, 1].
    """
    check_image_shape(image)

    dev, dtyp = image.device, image.dtype

    # -- Apply physical constraints (Eq. 13) --
    tau   = 0.3 + F.softplus(tau_raw.to(dev, dtyp))    # shadow power   > 0.3
    eta   = 0.3 + F.softplus(eta_raw.to(dev, dtyp))    # highlight power > 0.3
    xi    = torch.sigmoid(xi_raw.to(dev, dtyp))         # inflection      ∈ (0, 1)
    gamma = 0.1 + F.softplus(gamma_raw.to(dev, dtyp))  # gamma           > 0.1

    # Clamp input to [0, 1] (linear radiance can exceed 1 for HDR; CRF operates on [0,1])
    x = image.clamp(0.0, 1.0)

    # -- Eq. (15): Continuity coefficients --
    # a ensures C¹ continuity at the inflection point xi.
    lerp = tau * (1.0 - xi) + eta * xi   # = tau + xi*(eta - tau)
    a = (eta * xi) / lerp.clamp(min=1e-12)
    b = 1.0 - a

    # Reshape per-channel params for broadcasting over (..., H, W, 3)
    ndim = image.ndim
    spatial_ones = [1] * (ndim - 1)
    def _expand(t: Tensor) -> Tensor:
        return t.view(*spatial_ones, 3)

    tau_v   = _expand(tau)
    eta_v   = _expand(eta)
    xi_v    = _expand(xi)
    a_v     = _expand(a)
    b_v     = _expand(b)
    gamma_v = _expand(gamma)

    eps = 1e-6
    mask_low = x <= xi_v

    # -- Eq. (14): Piecewise S-curve --
    y_low  = a_v * torch.pow((x / xi_v.clamp(min=eps)).clamp(0.0, 1.0), tau_v)
    y_high = 1.0 - b_v * torch.pow(
        ((1.0 - x) / (1.0 - xi_v).clamp(min=eps)).clamp(0.0, 1.0), eta_v
    )

    y = torch.where(mask_low, y_low, y_high)

    # -- Eq. (16): Gamma correction --
    return torch.pow(y.clamp(min=eps), gamma_v)


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------


def apply_pipeline(
    image: Tensor,
    exposure_offset: float | Tensor = 0.0,
    vignetting_alpha: Tensor | None = None,
    vignetting_center: Tensor | None = None,
    color_offsets: dict[str, Tensor] | None = None,
    crf_tau_raw: Tensor | None = None,
    crf_eta_raw: Tensor | None = None,
    crf_xi_raw: Tensor | None = None,
    crf_gamma_raw: Tensor | None = None,
) -> Tensor:
    """Apply the full four-stage PPISP pipeline functionally.

    Stages are applied in the physically correct order:
    Exposure → Vignetting → Color Correction → CRF.

    Any stage whose parameters are ``None`` is skipped (identity).

    Args:
        image: Linear radiance image ``(H, W, 3)`` or ``(B, H, W, 3)``.
        exposure_offset: Δt in EV (default 0.0 = no change).
        vignetting_alpha: ``(3, 3)`` polynomial coefficients or None.
        vignetting_center: ``(2,)`` optical center or None.
        color_offsets: Dict ``{R, G, B, W} → (2,)`` or None.
        crf_tau_raw / crf_eta_raw / crf_xi_raw / crf_gamma_raw:
            Raw CRF parameters ``(3,)`` each, or None to skip CRF.

    Returns:
        Processed image tensor of the same shape as *image*.
    """
    check_image_shape(image)
    x = image

    # Stage 1: Exposure
    x = apply_exposure(x, exposure_offset)

    # Stage 2: Vignetting
    if vignetting_alpha is not None and vignetting_center is not None:
        x = apply_vignetting(x, vignetting_alpha, vignetting_center)

    # Stage 3: Color Correction
    if color_offsets is not None:
        x = apply_color_correction(x, color_offsets)

    # Stage 4: CRF
    if all(
        t is not None
        for t in (crf_tau_raw, crf_eta_raw, crf_xi_raw, crf_gamma_raw)
    ):
        x = apply_crf(x, crf_tau_raw, crf_eta_raw, crf_xi_raw, crf_gamma_raw)

    return x

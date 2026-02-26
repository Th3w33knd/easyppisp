"""
Internal implementation of the Chromaticity Homography (Section 4.3 of the PPISP paper).

Maps latent chromaticity offsets to a 3×3 homography H applied in RGI space,
enabling differentiable white-balance and color-correction.

Equation references (PPISP paper):
  - Latent → real offsets via ZCA preconditioning: Section B.1
  - Source chromaticities (R, G, B, W primaries):  Eq. (9)
  - Skew-symmetric cross-product matrix M:          Eq. (10)
  - Nullspace vector k from cross products:         Eq. (11)
  - Homography construction and normalization:       Eq. (12)
  - RGI intensity normalization:                     Eq. (7)
  - Chromaticity transform in RGI space:             Eq. (8)

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# ZCA Preconditioning Block-Diagonal Matrix (Section B.1)
# ---------------------------------------------------------------------------
# Each 2×2 block is the pseudoinverse of the ZCA-whitening Jacobian for one
# color class.  Stored as a constant 8×8 block-diagonal so that the latent
# offset vector (8-dim) can be mapped to real chromaticity offsets in a
# single matrix multiply.
#
# Block ordering matches the expected latent input ordering: B, R, G, W
# (Blue → Red → Green → Neutral/White).
#
# These values are copied verbatim from ppisp/__init__.py (_COLOR_PINV_BLOCK_DIAG).
_COLOR_PINV_BLOCK_DIAG: Tensor = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),  # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),  # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),  # Green
    torch.tensor([[0.0128369, -0.0034654], [-0.0034654, 0.0128158]]),  # Neutral/White
).to(torch.float32)

# ---------------------------------------------------------------------------
# Source chromaticity constants (Eq. 9)
# ---------------------------------------------------------------------------
# Chromaticities are expressed in homogeneous RG space: [r, g, 1]
# where r = R/(R+G+B) and g = G/(R+G+B).
#
# Pure Red pixel:    r=1, g=0  → [1, 0, 1]
# Pure Green pixel:  r=0, g=1  → [0, 1, 1]
# Pure Blue pixel:   r=0, g=0  → [0, 0, 1]
# Neutral white:     r=g=1/3   → [1/3, 1/3, 1]
_SRC_B = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
_SRC_R = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
_SRC_G = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
_SRC_W = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)

# Inverse of the source matrix S = [s_B | s_R | s_G] (column vectors).
# S = [[0, 1, 0],
#      [0, 0, 1],
#      [1, 1, 1]]
# S_inv computed analytically; det(S) = 1.
_S_INV = torch.tensor(
    [[-1.0, -1.0, 1.0],
     [ 1.0,  0.0, 0.0],
     [ 0.0,  1.0, 0.0]],
    dtype=torch.float32,
)

# Small epsilon for homography normalization — scaled to float32 precision.
_EPS_H = 1e-7   # for H[2,2] normalization (prevents divide-by-zero)
_EPS_I = 1e-5   # for intensity normalization (Eq. 7)


def build_homography(latent_offsets: Tensor) -> Tensor:
    """Construct the 3×3 chromaticity homography H from latent offsets.

    Implements Eq. (9)–(12) of the PPISP paper.

    The *latent_offsets* tensor must follow **B, R, G, W** ordering:
    ``[B_r, B_g, R_r, R_g, G_r, G_g, W_r, W_g]``.

    Args:
        latent_offsets: (8,) float tensor of chromaticity offsets in latent ZCA space.

    Returns:
        H: (3, 3) homography matrix mapping source → target chromaticities.

    Example:
        >>> H = build_homography(torch.zeros(8))  # identity-like
        >>> # H should be close to I (identity homography)
    """
    device = latent_offsets.device
    dtype = latent_offsets.dtype

    # -- Section B.1: Map latent offsets → real chromaticity offsets via ZCA pinv --
    # real_offsets = latent_offsets @ M  where M is 8×8 block-diagonal
    real_offsets = latent_offsets @ _COLOR_PINV_BLOCK_DIAG.to(device=device, dtype=dtype)

    # Unpack: ordering B, R, G, W (matches _COLOR_PINV_BLOCK_DIAG block order)
    bd = real_offsets[0:2]   # Blue chromaticity delta  [Δr, Δg]
    rd = real_offsets[2:4]   # Red chromaticity delta
    gd = real_offsets[4:6]   # Green chromaticity delta
    nd = real_offsets[6:8]   # Neutral/White chromaticity delta

    # -- Eq. (9): Compute target chromaticities t_k = s_k + Δc_k --
    src_b = _SRC_B.to(device=device, dtype=dtype)
    src_r = _SRC_R.to(device=device, dtype=dtype)
    src_g = _SRC_G.to(device=device, dtype=dtype)
    src_w = _SRC_W.to(device=device, dtype=dtype)

    t_b = torch.stack([src_b[0] + bd[0], src_b[1] + bd[1], src_b[2]])
    t_r = torch.stack([src_r[0] + rd[0], src_r[1] + rd[1], src_r[2]])
    t_g = torch.stack([src_g[0] + gd[0], src_g[1] + gd[1], src_g[2]])
    t_w = torch.stack([src_w[0] + nd[0], src_w[1] + nd[1], src_w[2]])

    # -- Target matrix T = [t_B | t_R | t_G] as columns --
    T = torch.stack([t_b, t_r, t_g], dim=1)   # (3, 3)

    # -- Eq. (10): Skew-symmetric cross-product matrix [t_W]_× applied to T --
    # [t_W]_× = [[0, -t_W[2], t_W[1]], [t_W[2], 0, -t_W[0]], [-t_W[1], t_W[0], 0]]
    zero = torch.zeros_like(t_w[0])
    skew = torch.stack([
        torch.stack([ zero,    -t_w[2],  t_w[1]]),
        torch.stack([ t_w[2],   zero,   -t_w[0]]),
        torch.stack([-t_w[1],  t_w[0],   zero  ]),
    ])  # (3, 3)
    M = skew @ T   # (3, 3)

    # -- Eq. (11): Nullspace vector k via pairwise row cross-products --
    # Select the most numerically stable pair (largest ‖·‖²)
    r0, r1, r2 = M[0], M[1], M[2]
    lam01 = torch.linalg.cross(r0, r1)
    lam02 = torch.linalg.cross(r0, r2)
    lam12 = torch.linalg.cross(r1, r2)

    n01 = (lam01 ** 2).sum()
    n02 = (lam02 ** 2).sum()
    n12 = (lam12 ** 2).sum()

    k = torch.where(
        n01 >= n02,
        torch.where(n01 >= n12, lam01, lam12),
        torch.where(n02 >= n12, lam02, lam12),
    )  # (3,)

    # -- Eq. (12): Construct and normalize H --
    s_inv = _S_INV.to(device=device, dtype=dtype)
    H = T @ torch.diag(k) @ s_inv   # (3, 3)
    H = H / (H[2, 2] + _EPS_H)      # Normalize so H[2,2] = 1

    return H


def apply_homography(image: Tensor, H: Tensor) -> Tensor:
    """Apply a 3×3 chromaticity homography to an image in RGI space.

    Implements Eq. (7)–(8) of the PPISP paper.

    The operation:
      1. Converts RGB → RGI (where I = R + G + B)
      2. Applies H in homogeneous RGI space
      3. Re-normalizes intensity to decouple color from exposure (Eq. 7)
      4. Converts RGI back to RGB

    Works for any leading batch dimensions: (H, W, 3) or (B, H, W, 3).

    Args:
        image: Input RGB image ``[..., 3]`` in linear radiance space.
        H: (3, 3) homography matrix from :func:`build_homography`.

    Returns:
        Color-corrected RGB image of the same shape as *image*.
    """
    orig_shape = image.shape
    H_dev = H.to(device=image.device, dtype=image.dtype)

    # -- Eq. (8): Convert RGB → RGI --
    intensity = image.sum(dim=-1, keepdim=True)   # I = R + G + B
    rgi = torch.cat([image[..., :2], intensity], dim=-1)   # [..., 3]

    # Flatten to (-1, 3) for the matrix multiply
    rgi_flat = rgi.reshape(-1, 3)             # (N, 3)
    rgi_mapped = (H_dev @ rgi_flat.T).T       # (N, 3)

    # -- Eq. (7): Intensity normalization (decouples color correction from exposure) --
    # Scale mapped RGI so that the homogeneous coordinate (I') matches original I.
    src_intensity = rgi_flat[:, 2]            # original I
    dst_intensity = rgi_mapped[:, 2]          # mapped I'
    scale = src_intensity / (dst_intensity + _EPS_I)   # (N,)
    rgi_mapped = rgi_mapped * scale.unsqueeze(-1)

    rgi_mapped = rgi_mapped.reshape(orig_shape)

    # -- Convert RGI back to RGB: B = I - R - G --
    r_out = rgi_mapped[..., 0]
    g_out = rgi_mapped[..., 1]
    b_out = rgi_mapped[..., 2] - r_out - g_out

    return torch.stack([r_out, g_out, b_out], dim=-1)

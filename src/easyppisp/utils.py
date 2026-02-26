"""
I/O helpers and color space conversion utilities.

Provides explicit, never-silently-converting transformations between:
  - Linear radiance float tensors (what PPISP expects)
  - sRGB uint8/float images (what cameras/screens produce)
  - PIL Image objects (for convenient file I/O)
  - CHW / HWC layout conversions (for torchvision compatibility)

All functions raise :class:`~easyppisp.validation.PPISPShapeError` or
:class:`~easyppisp.validation.PPISPValueError` on incompatible input rather
than silently converting.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from .validation import PPISPShapeError, PPISPValueError

logger = logging.getLogger("easyppisp")

try:
    from PIL import Image as _PIL_Image   # type: ignore[import]
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Color Space Conversions
# ---------------------------------------------------------------------------


def srgb_to_linear(image: Tensor) -> Tensor:
    """Convert an sRGB image to linear radiance (inverse gamma / EOTF).

    Applies the standard IEC 61966-2-1 piecewise inverse:
      - For values ≤ 0.04045: ``linear = srgb / 12.92``
      - For values  > 0.04045: ``linear = ((srgb + 0.055) / 1.055) ^ 2.4``

    Args:
        image: sRGB image tensor in [0, 1]. Any shape.

    Returns:
        Linear radiance tensor of the same shape and dtype.

    Example:
        >>> linear = srgb_to_linear(torch.tensor([0.5]))
        >>> linear.item()  # ≈ 0.2140
    """
    image = image.clamp(0.0, 1.0)
    mask = image <= 0.04045
    out = torch.empty_like(image)
    out[mask] = image[mask] / 12.92
    out[~mask] = torch.pow((image[~mask] + 0.055) / 1.055, 2.4)
    return out


def linear_to_srgb(image: Tensor) -> Tensor:
    """Convert a linear radiance tensor to sRGB (gamma / OETF).

    Applies the standard IEC 61966-2-1 piecewise forward:
      - For values ≤ 0.0031308: ``srgb = linear * 12.92``
      - For values  > 0.0031308: ``srgb = 1.055 * linear^(1/2.4) - 0.055``

    Args:
        image: Linear radiance tensor in [0, 1]. Any shape.

    Returns:
        sRGB tensor of the same shape and dtype.
    """
    image = image.clamp(0.0, 1.0)
    mask = image <= 0.0031308
    out = torch.empty_like(image)
    out[mask] = image[mask] * 12.92
    out[~mask] = 1.055 * torch.pow(image[~mask].clamp(min=1e-10), 1.0 / 2.4) - 0.055
    return out


# ---------------------------------------------------------------------------
# dtype / range conversions
# ---------------------------------------------------------------------------


def from_uint8(image: Tensor) -> Tensor:
    """Convert a ``uint8`` [0, 255] tensor to ``float32`` [0, 1].

    Args:
        image: ``torch.uint8`` tensor of any shape.

    Returns:
        ``float32`` tensor in [0, 1] of the same shape.

    Raises:
        PPISPValueError: If the input dtype is not ``uint8``.
    """
    if image.dtype != torch.uint8:
        raise PPISPValueError(
            f"Expected a uint8 tensor, got {image.dtype}. "
            "Pass the raw byte tensor directly, or use `.float() / 255.0` manually."
        )
    return image.to(torch.float32) / 255.0


def to_uint8(image: Tensor) -> Tensor:
    """Convert a ``float32`` [0, 1] tensor to ``uint8`` [0, 255].

    Values are clamped to [0, 1] before conversion to prevent wrapping.

    Args:
        image: ``float`` tensor in [0, 1] of any shape.

    Returns:
        ``uint8`` tensor in [0, 255] of the same shape.
    """
    return (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


# ---------------------------------------------------------------------------
# Layout conversions (HWC ↔ CHW)
# ---------------------------------------------------------------------------


def hwc_to_chw(image: Tensor) -> Tensor:
    """Permute from channels-last ``(H, W, C)`` / ``(B, H, W, C)`` to channels-first.

    PPISP uses HWC (channels-last).  Use this when passing to torchvision
    or other CHW-first libraries.

    Args:
        image: ``(H, W, C)`` or ``(B, H, W, C)`` tensor.

    Returns:
        ``(C, H, W)`` or ``(B, C, H, W)`` tensor (no copy if layout allows).

    Raises:
        PPISPShapeError: If ndim is not 3 or 4.
    """
    if image.ndim == 3:
        return image.permute(2, 0, 1).contiguous()
    elif image.ndim == 4:
        return image.permute(0, 3, 1, 2).contiguous()
    raise PPISPShapeError(
        f"hwc_to_chw expects a 3D (H,W,C) or 4D (B,H,W,C) tensor, got shape {tuple(image.shape)}"
    )


def chw_to_hwc(image: Tensor) -> Tensor:
    """Permute from channels-first ``(C, H, W)`` / ``(B, C, H, W)`` to channels-last.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)`` tensor.

    Returns:
        ``(H, W, C)`` or ``(B, H, W, C)`` tensor.

    Raises:
        PPISPShapeError: If ndim is not 3 or 4.
    """
    if image.ndim == 3:
        return image.permute(1, 2, 0).contiguous()
    elif image.ndim == 4:
        return image.permute(0, 2, 3, 1).contiguous()
    raise PPISPShapeError(
        f"chw_to_hwc expects a 3D (C,H,W) or 4D (B,C,H,W) tensor, got shape {tuple(image.shape)}"
    )


# ---------------------------------------------------------------------------
# PIL I/O helpers
# ---------------------------------------------------------------------------


def from_pil(
    image: "_PIL_Image.Image",
    device: str = "cpu",
    linearize: bool = True,
) -> Tensor:
    """Load a PIL Image into a float32 HWC tensor.

    By default, converts ``RGB uint8 [0,255]`` → ``float32 [0,1]`` → ``linear radiance``
    using the standard sRGB inverse gamma.  Set ``linearize=False`` when the image is
    already in linear-light space (e.g., a 16-bit linear TIFF loaded via PIL) to avoid
    double-applying the inverse gamma.

    Args:
        image: PIL ``Image`` object (will be converted to RGB if needed).
        device: Target device string (e.g., ``'cpu'``, ``'cuda'``).
        linearize: If ``True`` (default), apply sRGB inverse gamma to convert to
            linear radiance. Set ``False`` for images already in linear space.

    Returns:
        ``(H, W, 3)`` float32 tensor.

    Raises:
        ImportError: If Pillow is not installed.
    """
    if not HAS_PIL:
        raise ImportError(
            "Pillow is required for from_pil(). Install with: pip install Pillow"
        )
    import numpy as np

    arr = torch.from_numpy(np.asarray(image.convert("RGB"))).to(device)   # uint8 HWC
    float_t = from_uint8(arr)
    return srgb_to_linear(float_t) if linearize else float_t


def to_pil(image: Tensor) -> "_PIL_Image.Image":
    """Convert a linear radiance HWC tensor to a PIL Image (sRGB uint8).

    Applies the sRGB forward gamma before converting to ``uint8``.

    Args:
        image: ``(H, W, 3)`` float32 linear radiance tensor.

    Returns:
        ``PIL.Image.Image`` in RGB mode.

    Raises:
        ImportError: If Pillow is not installed.
    """
    if not HAS_PIL:
        raise ImportError(
            "Pillow is required for to_pil(). Install with: pip install Pillow"
        )
    import numpy as np
    from PIL import Image

    srgb = linear_to_srgb(image)
    arr = to_uint8(srgb).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def load_image(path: str | Path, device: str = "cpu") -> Tensor:
    """Load an image file as a linear radiance HWC float32 tensor.

    Convenience wrapper around :func:`from_pil`.

    Args:
        path: File path to load.
        device: Target device.

    Returns:
        ``(H, W, 3)`` float32 linear radiance tensor.

    Raises:
        ImportError: If Pillow is not installed.
    """
    if not HAS_PIL:
        raise ImportError(
            "Pillow is required for load_image(). Install with: pip install Pillow"
        )
    from PIL import Image

    return from_pil(Image.open(path), device=device)


def save_image(image: Tensor, path: str | Path) -> None:
    """Save a linear radiance HWC tensor to an image file.

    Applies sRGB gamma before saving.

    Args:
        image: ``(H, W, 3)`` float32 linear radiance tensor.
        path: Output file path (format inferred from extension).

    Raises:
        ImportError: If Pillow is not installed.
    """
    if not HAS_PIL:
        raise ImportError(
            "Pillow is required for save_image(). Install with: pip install Pillow"
        )
    pil_img = to_pil(image)
    pil_img.save(str(path))
    logger.debug("Saved image to %s", path)

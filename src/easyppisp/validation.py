"""
Input validation utilities and custom exception types for easyppisp.

All public-facing functions call these validators at module boundaries
to provide clear, actionable error messages.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Custom Exception Classes
# ---------------------------------------------------------------------------


class PPISPValueError(ValueError):
    """Raised when a parameter value is physically implausible or out of range."""


class PPISPShapeError(ValueError):
    """Raised when a tensor has an incorrect shape for the requested operation."""


class PPISPDeviceError(RuntimeError):
    """Raised when tensors are on incompatible devices."""


class PPISPPhysicsWarning(UserWarning):
    """Emitted when usage deviates from the physically-plausible model assumptions."""


# ---------------------------------------------------------------------------
# Shape Validators
# ---------------------------------------------------------------------------


def check_image_shape(image: Tensor, name: str = "image") -> None:
    """Validate that *image* has shape ``(H, W, 3)`` or ``(B, H, W, 3)``.

    Args:
        image: Tensor to validate.
        name: Variable name used in error messages.

    Raises:
        PPISPShapeError: If the shape is not compatible.
    """
    if image.ndim not in (3, 4):
        raise PPISPShapeError(
            f"'{name}' must be (H, W, 3) [single image] or (B, H, W, 3) [batch]. "
            f"Got shape {tuple(image.shape)} (ndim={image.ndim}). "
            "If your tensor is CHW, use `easyppisp.utils.chw_to_hwc()` first."
        )
    if image.shape[-1] != 3:
        raise PPISPShapeError(
            f"'{name}' must have 3 channels (RGB) in the last dimension. "
            f"Got {image.shape[-1]} channels in shape {tuple(image.shape)}."
        )


def check_same_device(*tensors: Tensor, names: list[str] | None = None) -> None:
    """Validate that all tensors are on the same device.

    Args:
        *tensors: Tensors to compare.
        names: Optional names for error messages.

    Raises:
        PPISPDeviceError: If tensors are on different devices.
    """
    if len(tensors) < 2:
        return
    ref = tensors[0].device
    for i, t in enumerate(tensors[1:], 1):
        if t.device != ref:
            n0 = names[0] if names else "tensor[0]"
            ni = names[i] if names else f"tensor[{i}]"
            raise PPISPDeviceError(
                f"'{n0}' is on {ref} but '{ni}' is on {t.device}. "
                "All tensors must be on the same device."
            )


def check_exposure_range(delta_t: float | Tensor) -> None:
    """Warn if the exposure offset looks physically implausible.

    Args:
        delta_t: Exposure offset in EV (stops).
    """
    val = float(delta_t) if not isinstance(delta_t, Tensor) else delta_t.item()
    if abs(val) > 10.0:
        warnings.warn(
            f"Exposure offset Δt={val:.1f} EV is unusually large. "
            "Typical photographic range is [-3, +3] EV. "
            "Did you pass a linear multiplier instead of a log₂ value? "
            "Hint: delta_t = log2(desired_multiplier).",
            PPISPPhysicsWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Value / Content Validators
# ---------------------------------------------------------------------------


def check_linear_radiance(image: Tensor, enforce: bool = False) -> None:
    """Warn if the image appears to be in uint8 [0, 255] space.

    Uses a conservative heuristic: warns only if ``max > 10.0``, which is a
    safe indicator that the values are uint8 integers rather than normalized
    float radiance (which can exceed 1.0 for HDR content, but rarely by 10×).

    Args:
        image: Tensor to inspect.
        enforce: If True, raises ``PPISPValueError`` instead of warning.

    Raises:
        PPISPValueError: Only when *enforce=True* and the check triggers.
    """
    if image.max() > 10.0:
        msg = (
            f"Image maximum is {image.max().item():.1f}, which suggests a uint8 [0, 255] tensor. "
            "PPISP expects normalized linear radiance float tensors in [0, ~1] range. "
            "Convert with `easyppisp.utils.from_uint8(image)` first."
        )
        if enforce:
            raise PPISPValueError(msg)
        warnings.warn(msg, PPISPPhysicsWarning, stacklevel=3)

"""
easyppisp — A developer-friendly wrapper for Physically-Plausible ISP (PPISP).

Provides a clean, composable, differentiable API over NVIDIA's PPISP library
for simulating real camera physics:

  Exposure → Vignetting → Color Correction → Camera Response Function

Quickstart (≤ 5 lines):
    >>> from easyppisp.utils import load_image, save_image
    >>> import easyppisp
    >>> image = load_image("photo.jpg")               # linear float32 HWC
    >>> bright = easyppisp.apply(image, exposure=1.5) # +1.5 stops
    >>> save_image(bright, "bright.jpg")

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

# -- Core data structures --
from .params import PipelineParams, PipelineResult

# -- Functional API (stateless, differentiable) --
from .functional import (
    apply_exposure,
    apply_vignetting,
    apply_color_correction,
    apply_crf,
    apply_pipeline,
)

# -- Module API (learnable nn.Module wrappers) --
from .modules import (
    ExposureOffset,
    Vignetting,
    ColorCorrection,
    CameraResponseFunction,
    ISPPipeline,
    ISPController,
)

# -- Task API (high-level workflows) --
from .tasks import CameraSimulator, PhysicalAugmentation, CameraMatchPair

# -- Regularization Losses --
from . import losses

# -- Presets --
from .presets import FilmPreset, load_preset

# -- Utilities --
from .utils import (
    srgb_to_linear,
    linear_to_srgb,
    from_uint8,
    to_uint8,
    hwc_to_chw,
    chw_to_hwc,
)

# -- Validation / exceptions (re-exported for user catch clauses) --
from .validation import (
    PPISPValueError,
    PPISPShapeError,
    PPISPDeviceError,
    PPISPPhysicsWarning,
)

# Set up a library-level NullHandler so that downstream users can configure
# logging without getting spurious "no handler found" warnings.
logging.getLogger("easyppisp").addHandler(logging.NullHandler())

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__: str = _pkg_version("easyppisp")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:
    __version__ = "0.0.0"


# ---------------------------------------------------------------------------
# Convenience one-liner
# ---------------------------------------------------------------------------


def apply(
    image: Tensor,
    exposure: float = 0.0,
    preset: str | None = None,
    vignetting_alpha: "Tensor | None" = None,
    vignetting_center: "Tensor | None" = None,
    color_offsets: "dict | None" = None,
    crf_tau_raw: "Tensor | None" = None,
    crf_eta_raw: "Tensor | None" = None,
    crf_xi_raw: "Tensor | None" = None,
    crf_gamma_raw: "Tensor | None" = None,
) -> Tensor:
    """Quickly apply ISP adjustments to a linear radiance image.

    For simple exposure-only adjustments this is a one-liner.
    Pass *preset* to run a full named pipeline instead.
    Any additional stage parameters are forwarded explicitly — there are no
    silent kwargs; unknown arguments cause a ``TypeError`` at call time.

    Args:
        image: Linear radiance image ``(H, W, 3)`` or ``(B, H, W, 3)``.
        exposure: Exposure offset in EV stops. Default 0.0 = no change.
        preset: Optional named preset (see :func:`~easyppisp.presets.FilmPreset.list_presets`).
            If provided, runs the full preset pipeline and *exposure* overrides the
            preset's exposure stage. All other stage parameters are ignored.
        vignetting_alpha: ``(3, 3)`` polynomial coefficients or None (skip stage).
        vignetting_center: ``(2,)`` optical center or None (skip stage).
        color_offsets: Dict ``{R, G, B, W} -> (2,)`` or None (skip stage).
        crf_tau_raw / crf_eta_raw / crf_xi_raw / crf_gamma_raw:
            Raw CRF parameters ``(3,)`` each.  All four must be provided together,
            or all must be None (skip stage).

    Returns:
        Processed image tensor of the same shape as *image*.

    Example:
        >>> result = easyppisp.apply(img, exposure=1.5)
        >>> result = easyppisp.apply(img, preset="kodak_portra_400", exposure=-0.5)
        >>> result = easyppisp.apply(img, exposure=0.5, color_offsets=my_offsets)
    """
    if preset is not None:
        cam = CameraSimulator(preset=preset)
        cam.set_exposure(exposure)
        return cam(image)
    return apply_pipeline(
        image,
        exposure_offset=exposure,
        vignetting_alpha=vignetting_alpha,
        vignetting_center=vignetting_center,
        color_offsets=color_offsets,
        crf_tau_raw=crf_tau_raw,
        crf_eta_raw=crf_eta_raw,
        crf_xi_raw=crf_xi_raw,
        crf_gamma_raw=crf_gamma_raw,
    )


__all__ = [
    # version
    "__version__",
    # convenience
    "apply",
    # data structures
    "PipelineParams",
    "PipelineResult",
    # functional
    "apply_exposure",
    "apply_vignetting",
    "apply_color_correction",
    "apply_crf",
    "apply_pipeline",
    # modules
    "ExposureOffset",
    "Vignetting",
    "ColorCorrection",
    "CameraResponseFunction",
    "ISPPipeline",
    "ISPController",
    # tasks
    "CameraSimulator",
    "PhysicalAugmentation",
    "CameraMatchPair",
    # losses
    "losses",
    # presets
    "FilmPreset",
    "load_preset",
    # utils
    "srgb_to_linear",
    "linear_to_srgb",
    "from_uint8",
    "to_uint8",
    "hwc_to_chw",
    "chw_to_hwc",
    # exceptions
    "PPISPValueError",
    "PPISPShapeError",
    "PPISPDeviceError",
    "PPISPPhysicsWarning",
]

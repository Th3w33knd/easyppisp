"""
PyTorch ``nn.Module`` wrappers for ISP optimization.

Each stage of the PPISP pipeline is a standalone, independently usable module:
  - :class:`ExposureOffset`         — learnable Δt (per-frame exposure)
  - :class:`Vignetting`             — learnable polynomial falloff + optical center
  - :class:`ColorCorrection`        — learnable chromaticity homography offsets
  - :class:`CameraResponseFunction` — learnable piecewise-power S-curve + gamma
  - :class:`ISPPipeline`            — composable sequence of any of the above

All modules expose:
  - ``.forward(x)`` — differentiable forward pass
  - ``.get_params_dict()`` — human-readable parameter snapshot
  - ``.from_params(p)`` classmethod — construct from a :class:`~easyppisp.params.PipelineParams`

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import torch
from torch import nn, Tensor

from .functional import (
    apply_exposure,
    apply_vignetting,
    apply_color_correction,
    apply_crf,
)
from .params import PipelineParams, PipelineResult
from .validation import PPISPPhysicsWarning

logger = logging.getLogger("easyppisp")


# ---------------------------------------------------------------------------
# ExposureOffset
# ---------------------------------------------------------------------------


class ExposureOffset(nn.Module):
    """Learnable per-frame exposure offset (Eq. 3).

    Args:
        delta_t: Initial exposure offset in EV (stops). Default 0.0 = no change.

    Example:
        >>> mod = ExposureOffset(delta_t=1.0)
        >>> out = mod(torch.ones(4, 4, 3))   # 2× brighter
    """

    def __init__(self, delta_t: float = 0.0) -> None:
        super().__init__()
        self.delta_t = nn.Parameter(torch.tensor(delta_t, dtype=torch.float32))

    @classmethod
    def from_params(cls, params: PipelineParams) -> "ExposureOffset":
        """Construct from a :class:`~easyppisp.params.PipelineParams`."""
        return cls(delta_t=params.exposure_offset)

    def forward(self, x: Tensor) -> Tensor:
        return apply_exposure(x, self.delta_t)

    def get_params_dict(self) -> dict:
        """Return human-readable parameter values."""
        return {"exposure_offset_ev": self.delta_t.item()}

    def __repr__(self) -> str:
        return f"ExposureOffset(delta_t={self.delta_t.item():.4f})"


# ---------------------------------------------------------------------------
# Vignetting
# ---------------------------------------------------------------------------


class Vignetting(nn.Module):
    """Learnable per-camera chromatic vignetting (Eq. 4–5).

    Args:
        alpha: ``(3, 3)`` initial polynomial coefficients ``[channel, term]``.
            Identity = all zeros.
        center: ``(2,)`` initial optical center offset (normalized coords).
            Identity = ``[0.0, 0.0]`` (image center).

    Example:
        >>> vig = Vignetting()
        >>> out = vig(torch.ones(64, 64, 3))   # identity at default params
    """

    def __init__(
        self,
        alpha: Tensor | None = None,
        center: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(
            alpha.clone().float() if alpha is not None else torch.zeros(3, 3)
        )
        self.center = nn.Parameter(
            center.clone().float() if center is not None else torch.zeros(2)
        )

    @classmethod
    def from_params(cls, params: PipelineParams) -> "Vignetting":
        return cls(alpha=params.vignetting_alpha, center=params.vignetting_center)

    def forward(self, x: Tensor) -> Tensor:
        return apply_vignetting(x, self.alpha, self.center)

    def get_params_dict(self) -> dict:
        return {
            "vignetting_alpha": self.alpha.detach().tolist(),
            "vignetting_center": self.center.detach().tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"Vignetting(alpha_mean={self.alpha.mean().item():.4f}, "
            f"center={self.center.detach().tolist()})"
        )


# ---------------------------------------------------------------------------
# ColorCorrection
# ---------------------------------------------------------------------------


class ColorCorrection(nn.Module):
    """Learnable per-frame chromaticity homography (Eq. 6–12).

    Stores four separate ``nn.Parameter`` tensors (B, R, G, W) so that
    each can be independently optimized and logged.

    Args:
        offsets: Dict ``{'R', 'G', 'B', 'W'} → (2,)`` initial offset tensors.
            Identity = all zeros.

    Example:
        >>> cc = ColorCorrection()
        >>> out = cc(torch.rand(8, 8, 3))
    """

    def __init__(self, offsets: dict[str, Tensor] | None = None) -> None:
        super().__init__()
        zeros = torch.zeros(2, dtype=torch.float32)
        if offsets is None:
            offsets = {}
        # Register per-channel parameters individually for clean state_dict keys
        self.b_off = nn.Parameter((offsets.get("B", zeros)).clone().float())
        self.r_off = nn.Parameter((offsets.get("R", zeros)).clone().float())
        self.g_off = nn.Parameter((offsets.get("G", zeros)).clone().float())
        self.w_off = nn.Parameter((offsets.get("W", zeros)).clone().float())

    @classmethod
    def from_params(cls, params: PipelineParams) -> "ColorCorrection":
        return cls(offsets=params.color_offsets)

    def forward(self, x: Tensor) -> Tensor:
        offsets = {"B": self.b_off, "R": self.r_off, "G": self.g_off, "W": self.w_off}
        return apply_color_correction(x, offsets)

    def get_params_dict(self) -> dict:
        return {
            "color_B": self.b_off.detach().tolist(),
            "color_R": self.r_off.detach().tolist(),
            "color_G": self.g_off.detach().tolist(),
            "color_W": self.w_off.detach().tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"ColorCorrection(R={self.r_off.detach().tolist()}, "
            f"G={self.g_off.detach().tolist()}, B={self.b_off.detach().tolist()}, "
            f"W={self.w_off.detach().tolist()})"
        )


# ---------------------------------------------------------------------------
# CameraResponseFunction
# ---------------------------------------------------------------------------


class CameraResponseFunction(nn.Module):
    """Learnable per-camera piecewise-power S-curve + gamma (Eq. 13–16).

    Stores *raw* (unconstrained) parameters; physical constraints
    (``softplus`` / ``sigmoid``) are applied inside :func:`apply_crf`.
    This guarantees monotonicity and numerical stability during optimization.

    Args:
        tau:   (3,) raw shadow power (default zeros → actual tau ≈ 1.0 after softplus+0.3).
        eta:   (3,) raw highlight power.
        xi:    (3,) raw inflection point (default zeros → xi = 0.5 after sigmoid).
        gamma: (3,) raw gamma (default zeros → gamma ≈ 0.8 after softplus+0.1).

    Example:
        >>> crf = CameraResponseFunction()
        >>> out = crf(torch.rand(8, 8, 3))
    """

    def __init__(
        self,
        tau: Tensor | None = None,
        eta: Tensor | None = None,
        xi: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> None:
        super().__init__()
        zeros = torch.zeros(3, dtype=torch.float32)
        self.tau   = nn.Parameter((tau   if tau   is not None else zeros).clone().float())
        self.eta   = nn.Parameter((eta   if eta   is not None else zeros).clone().float())
        self.xi    = nn.Parameter((xi    if xi    is not None else zeros).clone().float())
        self.gamma = nn.Parameter((gamma if gamma is not None else zeros).clone().float())

    @classmethod
    def from_params(cls, params: PipelineParams) -> "CameraResponseFunction":
        return cls(
            tau=params.crf_tau,
            eta=params.crf_eta,
            xi=params.crf_xi,
            gamma=params.crf_gamma,
        )

    def forward(self, x: Tensor) -> Tensor:
        return apply_crf(x, self.tau, self.eta, self.xi, self.gamma)

    def get_params_dict(self) -> dict:
        """Return both raw (optimizer) and physical (constrained) CRF values.

        The ``_raw`` keys hold the unconstrained ``nn.Parameter`` values suitable
        for round-tripping through ``PipelineParams``.  The ``_phys`` keys hold the
        human-readable constrained values for logging and inspection.
        """
        import torch.nn.functional as F
        tau_phys   = (0.3 + F.softplus(self.tau)).detach().tolist()
        eta_phys   = (0.3 + F.softplus(self.eta)).detach().tolist()
        xi_phys    = torch.sigmoid(self.xi).detach().tolist()
        gamma_phys = (0.1 + F.softplus(self.gamma)).detach().tolist()
        return {
            # Physical (human-readable) — for inspection / logging only
            "crf_tau_phys":   tau_phys,
            "crf_eta_phys":   eta_phys,
            "crf_xi_phys":    xi_phys,
            "crf_gamma_phys": gamma_phys,
            # Raw (unconstrained) — compatible with PipelineParams fields
            "crf_tau":   self.tau.detach().tolist(),
            "crf_eta":   self.eta.detach().tolist(),
            "crf_xi":    self.xi.detach().tolist(),
            "crf_gamma": self.gamma.detach().tolist(),
        }

    def __repr__(self) -> str:
        p = self.get_params_dict()
        return (
            f"CameraResponseFunction(tau={[round(v,3) for v in p['crf_tau']]}, "
            f"eta={[round(v,3) for v in p['crf_eta']]}, "
            f"xi={[round(v,3) for v in p['crf_xi']]}, "
            f"gamma={[round(v,3) for v in p['crf_gamma']]})"
        )


# ---------------------------------------------------------------------------
# ISPPipeline
# ---------------------------------------------------------------------------

# Physical ordering: linear ops must precede the non-linear CRF stage.
_PHYSICAL_ORDER = (ExposureOffset, Vignetting, ColorCorrection, CameraResponseFunction)
_LINEAR_MODULES = (ExposureOffset, Vignetting, ColorCorrection)


class ISPPipeline(nn.Module):
    """Composable sequence of ISP modules.

    Modules are applied in the order provided.  If a :class:`CameraResponseFunction`
    is followed by any linear module (exposure, vignetting, color), a
    :class:`~easyppisp.validation.PPISPPhysicsWarning` is emitted to alert
    the user about the non-physical ordering.

    By default, all four stages are included with identity parameters.

    Args:
        modules: Ordered sequence of ``nn.Module`` instances.
            Defaults to ``[ExposureOffset, Vignetting, ColorCorrection, CameraResponseFunction]``.

    Example:
        >>> pipeline = ISPPipeline([ExposureOffset(delta_t=1.0), CameraResponseFunction()])
        >>> result = pipeline(image, return_intermediates=True)
        >>> result.intermediates.keys()
        dict_keys(['ExposureOffset', 'CameraResponseFunction'])
    """

    def __init__(self, modules: Sequence[nn.Module] | None = None) -> None:
        super().__init__()
        if modules is None:
            modules = [
                ExposureOffset(),
                Vignetting(),
                ColorCorrection(),
                CameraResponseFunction(),
            ]
        self.pipeline = nn.ModuleList(modules)
        self._check_physical_ordering()

    @classmethod
    def from_params(cls, params: PipelineParams) -> "ISPPipeline":
        """Build a full four-stage pipeline from a :class:`~easyppisp.params.PipelineParams`.

        Args:
            params: Configuration for all four stages.

        Returns:
            Configured :class:`ISPPipeline` instance.
        """
        return cls([
            ExposureOffset.from_params(params),
            Vignetting.from_params(params),
            ColorCorrection.from_params(params),
            CameraResponseFunction.from_params(params),
        ])

    def _check_physical_ordering(self) -> None:
        """Warn if any linear module appears after CameraResponseFunction."""
        seen_crf = False
        for mod in self.pipeline:
            if isinstance(mod, CameraResponseFunction):
                seen_crf = True
            elif seen_crf and isinstance(mod, _LINEAR_MODULES):
                warnings.warn(
                    f"{type(mod).__name__} is placed after CameraResponseFunction. "
                    "CRF maps linear radiance to display-referred (sRGB) space. "
                    "Applying linear operations (exposure, vignetting, color correction) "
                    "after CRF breaks the physical model assumptions.",
                    PPISPPhysicsWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "Non-physical ISP ordering: %s after CameraResponseFunction.",
                    type(mod).__name__,
                )

    def forward(self, image: Tensor, return_intermediates: bool = False) -> PipelineResult:
        """Run the full pipeline.

        Args:
            image: Input image ``(H, W, 3)`` or ``(B, H, W, 3)``.
            return_intermediates: If True, collect per-stage outputs in
                :attr:`PipelineResult.intermediates` keyed by class name.

        Returns:
            :class:`~easyppisp.params.PipelineResult` with ``.final`` and
            optionally ``.intermediates``.
        """
        intermediates: dict[str, Tensor] = {}
        x = image
        for mod in self.pipeline:
            x = mod(x)
            if return_intermediates:
                intermediates[type(mod).__name__] = x.clone()

        return PipelineResult(
            final=x,
            intermediates=intermediates if return_intermediates else None,
        )

    def get_params_dict(self) -> dict:
        """Aggregate human-readable parameters from all modules."""
        out = {}
        for mod in self.pipeline:
            if hasattr(mod, "get_params_dict"):
                out[type(mod).__name__] = mod.get_params_dict()
        return out

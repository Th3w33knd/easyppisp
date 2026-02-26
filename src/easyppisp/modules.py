"""
PyTorch ``nn.Module`` wrappers for ISP optimization.

Each stage of the PPISP pipeline is a standalone, independently usable module:
  - :class:`ExposureOffset`         — learnable Δt (per-frame exposure)
  - :class:`Vignetting`             — learnable polynomial falloff + optical center
  - :class:`ColorCorrection`        — learnable chromaticity homography offsets
  - :class:`CameraResponseFunction` — learnable piecewise-power S-curve + gamma
  - :class:`ISPPipeline`            — composable sequence of any of the above
  - :class:`ISPController`          — predictive controller for novel views (Eq. 17)

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
    apply_pipeline,
)
from .params import (
    PipelineParams,
    PipelineResult,
    _CRF_TAU_IDENTITY,
    _CRF_ETA_IDENTITY,
    _CRF_XI_IDENTITY,
    _CRF_GAMMA_IDENTITY,
)
from .validation import PPISPPhysicsWarning, check_image_shape

logger = logging.getLogger("easyppisp")


# ---------------------------------------------------------------------------
# ExposureOffset
# ---------------------------------------------------------------------------


class ExposureOffset(nn.Module):
    """Learnable per-frame exposure offset (Eq. 3).

    Args:
        delta_t: Initial exposure offset in EV (stops). Default 0.0 = no change.
    """

    def __init__(self, delta_t: float = 0.0) -> None:
        super().__init__()
        self.delta_t = nn.Parameter(torch.tensor(delta_t, dtype=torch.float32))

    @classmethod
    def from_params(cls, params: PipelineParams) -> "ExposureOffset":
        return cls(delta_t=params.exposure_offset)

    def forward(self, x: Tensor) -> Tensor:
        return apply_exposure(x, self.delta_t)

    def get_params_dict(self) -> dict:
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
        center: ``(2,)`` initial optical center offset (normalized coords).
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
    """Learnable per-frame chromaticity homography (Eq. 6–12)."""

    def __init__(self, offsets: dict[str, Tensor] | None = None) -> None:
        super().__init__()
        zeros = torch.zeros(2, dtype=torch.float32)
        if offsets is None:
            offsets = {}
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
    """Learnable per-camera piecewise-power S-curve + gamma (Eq. 13–16)."""

    def __init__(
        self,
        tau: Tensor | None = None,
        eta: Tensor | None = None,
        xi: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> None:
        super().__init__()
        def _get(val, ident):
            if val is not None:
                return val.clone().float()
            return torch.full((3,), ident, dtype=torch.float32)

        self.tau   = nn.Parameter(_get(tau,   _CRF_TAU_IDENTITY))
        self.eta   = nn.Parameter(_get(eta,   _CRF_ETA_IDENTITY))
        self.xi    = nn.Parameter(_get(xi,    _CRF_XI_IDENTITY))
        self.gamma = nn.Parameter(_get(gamma, _CRF_GAMMA_IDENTITY))

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
        import torch.nn.functional as F
        tau_phys   = (0.3 + F.softplus(self.tau)).detach().tolist()
        eta_phys   = (0.3 + F.softplus(self.eta)).detach().tolist()
        xi_phys    = torch.sigmoid(self.xi).detach().tolist()
        gamma_phys = (0.1 + F.softplus(self.gamma)).detach().tolist()
        return {
            "crf_tau_phys": tau_phys,
            "crf_eta_phys": eta_phys,
            "crf_xi_phys": xi_phys,
            "crf_gamma_phys": gamma_phys,
            "crf_tau": self.tau.detach().tolist(),
            "crf_eta": self.eta.detach().tolist(),
            "crf_xi": self.xi.detach().tolist(),
            "crf_gamma": self.gamma.detach().tolist(),
        }


# ---------------------------------------------------------------------------
# ISPPipeline
# ---------------------------------------------------------------------------


_LINEAR_MODULES = (ExposureOffset, Vignetting, ColorCorrection)


class ISPPipeline(nn.Module):
    """Composable sequence of ISP modules.

    Default: Exposure → Vignetting → Color → CRF.
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
        return cls([
            ExposureOffset.from_params(params),
            Vignetting.from_params(params),
            ColorCorrection.from_params(params),
            CameraResponseFunction.from_params(params),
        ])

    def _check_physical_ordering(self) -> None:
        seen_crf = False
        for mod in self.pipeline:
            if isinstance(mod, CameraResponseFunction):
                seen_crf = True
            elif seen_crf and isinstance(mod, _LINEAR_MODULES):
                warnings.warn(
                    f"{type(mod).__name__} is placed after CameraResponseFunction.",
                    PPISPPhysicsWarning,
                    stacklevel=2,
                )

    def forward(self, image: Tensor, return_intermediates: bool = False) -> PipelineResult:
        intermediates: dict[str, Tensor] = {}
        x = image
        for mod in self.pipeline:
            x = mod(x)
            if return_intermediates:
                intermediates[type(mod).__name__] = x.clone()

        used = PipelineParams(
            exposure_offset=next((m.delta_t.item() for m in self.pipeline if isinstance(m, ExposureOffset)), 0.0),
            vignetting_alpha=next((m.alpha.detach() for m in self.pipeline if isinstance(m, Vignetting)), torch.zeros(3, 3)),
            vignetting_center=next((m.center.detach() for m in self.pipeline if isinstance(m, Vignetting)), torch.zeros(2)),
            color_offsets=next(({"R": m.r_off.detach(), "G": m.g_off.detach(), "B": m.b_off.detach(), "W": m.w_off.detach()} 
                               for m in self.pipeline if isinstance(m, ColorCorrection)), PipelineParams().color_offsets),
            crf_tau=next((m.tau.detach() for m in self.pipeline if isinstance(m, CameraResponseFunction)), torch.zeros(3)),
            crf_eta=next((m.eta.detach() for m in self.pipeline if isinstance(m, CameraResponseFunction)), torch.zeros(3)),
            crf_xi=next((m.xi.detach() for m in self.pipeline if isinstance(m, CameraResponseFunction)), torch.zeros(3)),
            crf_gamma=next((m.gamma.detach() for m in self.pipeline if isinstance(m, CameraResponseFunction)), torch.zeros(3)),
        )

        return PipelineResult(
            final=x,
            intermediates=intermediates if return_intermediates else None,
            params_used=used
        )

    def get_params_dict(self) -> dict:
        out = {}
        for mod in self.pipeline:
            if hasattr(mod, "get_params_dict"):
                out[type(mod).__name__] = mod.get_params_dict()
        return out


# ---------------------------------------------------------------------------
# ISPController
# ---------------------------------------------------------------------------


class ISPController(nn.Module):
    """CNN-based controller for predicting ISP parameters from images (Eq. 17)."""

    def __init__(
        self,
        cnn_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        pool_grid_size: tuple[int, int] = (5, 5),
    ) -> None:
        super().__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cnn_feature_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(pool_grid_size),
            nn.Flatten(),
        )

        cnn_output_dim = cnn_feature_dim * pool_grid_size[0] * pool_grid_size[1]
        input_dim = cnn_output_dim + 1  # +1 for optional prior_exposure

        layers = []
        curr_dim = input_dim
        for _ in range(num_mlp_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim
        self.mlp_trunk = nn.Sequential(*layers)

        self.exposure_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 8)

    def forward(
        self,
        image: Tensor,
        prior_exposure: float | Tensor = 0.0,
    ) -> dict[str, Tensor]:
        """Predict ISP parameters. Returns dict with 'exposure_offset' and 'color_params_flat'."""
        check_image_shape(image)
        if image.ndim == 3:
            x = image.permute(2, 0, 1).unsqueeze(0)
            if not isinstance(prior_exposure, Tensor):
                prior_exposure = torch.tensor([[float(prior_exposure)]], device=x.device, dtype=x.dtype)
            elif prior_exposure.ndim == 0:
                prior_exposure = prior_exposure.view(1, 1)
        else:
            x = image.permute(0, 3, 1, 2)
            if not isinstance(prior_exposure, Tensor):
                prior_exposure = torch.full((x.shape[0], 1), float(prior_exposure), device=x.device, dtype=x.dtype)
            elif prior_exposure.ndim == 1:
                prior_exposure = prior_exposure.unsqueeze(-1)

        feats = self.cnn_encoder(x)
        combined = torch.cat([feats, prior_exposure], dim=-1)
        hidden = self.mlp_trunk(combined)

        return {
            "exposure_offset": self.exposure_head(hidden),
            "color_params_flat": self.color_head(hidden),
        }

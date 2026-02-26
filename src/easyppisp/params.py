"""
Data structures for configuring and storing PPISP pipeline states.

Maps directly to the paper's physical model:
  - Exposure: Δt in EV stops (Eq. 3)
  - Vignetting: polynomial falloff coefficients + optical center (Eq. 4-5)
  - Color: chromaticity offsets as Tensors in B/R/G/W order (Eq. 6-12)
  - CRF: raw (unconstrained) parameters, constrained internally via softplus/sigmoid (Eq. 13-16)

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Pre-computed raw values that produce an identity (no-op) CRF after the
# physical constraints applied inside apply_crf():
#
#   tau_phys   = 0.3 + softplus(tau_raw)   → identity at tau_phys  = 1.0
#   eta_phys   = 0.3 + softplus(eta_raw)   → identity at eta_phys  = 1.0
#   xi_phys    = sigmoid(xi_raw)            → identity at xi_phys   = 0.5
#   gamma_phys = 0.1 + softplus(gamma_raw) → identity at gamma_phys = 1.0
#
# softplus_inv(y, min_val) = log(expm1(y - min_val))
# sigmoid_inv(0.5)         = 0.0
# ---------------------------------------------------------------------------
_CRF_TAU_IDENTITY   = math.log(math.expm1(0.7))   # ≈  0.013659
_CRF_ETA_IDENTITY   = math.log(math.expm1(0.7))   # ≈  0.013659
_CRF_XI_IDENTITY    = 0.0                           # sigmoid(0) = 0.5
_CRF_GAMMA_IDENTITY = math.log(math.expm1(0.9))   # ≈  0.378165


@dataclass
class PipelineParams:
    """All parameters for the full ISP pipeline, in human-readable form.

    Parameters are in their *raw* (unconstrained) representation where applicable.
    The functional API applies softplus/sigmoid constraints internally to guarantee
    physical plausibility (e.g., monotonic CRF).

    Color offsets use Tensor values to preserve gradient tracking through the pipeline.
    The internal ordering enforced when calling build_homography is always B, R, G, W.

    Attributes:
        exposure_offset: Δt in EV (exposure stops). 0.0 = no change, +1.0 = 2× brighter.
        vignetting_alpha: (3, 3) per-channel polynomial coefficients [channel, term].
            Channels = RGB, terms = α₁, α₂, α₃ in 1 + α₁r² + α₂r⁴ + α₃r⁶.
            Identity = all zeros.
        vignetting_center: (2,) optical center [cx, cy] in pixel coordinates relative to
            image center, divided by max(H, W). Identity = [0.0, 0.0].
        color_offsets: Dict mapping 'R', 'G', 'B', 'W' to (2,) chromaticity offset tensors.
            Identity = all zeros.
        crf_tau: (3,) raw shadow power per channel (constrained by softplus + 0.3 internally).
        crf_eta: (3,) raw highlight power per channel (constrained by softplus + 0.3 internally).
        crf_xi: (3,) raw inflection point per channel (constrained by sigmoid internally).
        crf_gamma: (3,) raw gamma per channel (constrained by softplus + 0.1 internally).
    """

    exposure_offset: float = 0.0

    vignetting_alpha: Tensor = field(
        default_factory=lambda: torch.zeros((3, 3), dtype=torch.float32)
    )
    vignetting_center: Tensor = field(
        default_factory=lambda: torch.zeros(2, dtype=torch.float32)
    )

    # Color offsets as Tensors to preserve gradient flow.
    # Keys: 'R', 'G', 'B', 'W' — internally reordered to B, R, G, W before homography build.
    color_offsets: dict[str, Tensor] = field(
        default_factory=lambda: {
            "R": torch.zeros(2, dtype=torch.float32),
            "G": torch.zeros(2, dtype=torch.float32),
            "B": torch.zeros(2, dtype=torch.float32),
            "W": torch.zeros(2, dtype=torch.float32),
        }
    )

    # Raw (unbounded) CRF parameters. Constraints applied inside apply_crf():
    #   tau   -> 0.3 + softplus(tau_raw)    (shadow power > 0.3)
    #   eta   -> 0.3 + softplus(eta_raw)    (highlight power > 0.3)
    #   xi    -> sigmoid(xi_raw)            (inflection in (0, 1))
    #   gamma -> 0.1 + softplus(gamma_raw)  (gamma > 0.1)
    # Default raw=0: tau=0.3+ln2≈1.0, eta≈1.0, xi=0.5, gamma≈0.8
    # Defaults are pre-computed raw values that yield an IDENTITY transform:
    #   tau_phys=1.0, eta_phys=1.0, xi_phys=0.5, gamma_phys=1.0
    # This ensures PipelineParams() / ISPPipeline() is a true no-op.
    crf_tau: Tensor = field(
        default_factory=lambda: torch.full((3,), _CRF_TAU_IDENTITY, dtype=torch.float32)
    )
    crf_eta: Tensor = field(
        default_factory=lambda: torch.full((3,), _CRF_ETA_IDENTITY, dtype=torch.float32)
    )
    crf_xi: Tensor = field(
        default_factory=lambda: torch.full((3,), _CRF_XI_IDENTITY, dtype=torch.float32)
    )
    crf_gamma: Tensor = field(
        default_factory=lambda: torch.full((3,), _CRF_GAMMA_IDENTITY, dtype=torch.float32)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "exposure_offset": self.exposure_offset,
            "vignetting_alpha": self.vignetting_alpha.tolist(),
            "vignetting_center": self.vignetting_center.tolist(),
            "color_offsets": {k: v.tolist() for k, v in self.color_offsets.items()},
            "crf_tau": self.crf_tau.tolist(),
            "crf_eta": self.crf_eta.tolist(),
            "crf_xi": self.crf_xi.tolist(),
            "crf_gamma": self.crf_gamma.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineParams":
        """Deserialize from a dictionary."""
        raw_color = d.get(
            "color_offsets",
            {"R": [0.0, 0.0], "G": [0.0, 0.0], "B": [0.0, 0.0], "W": [0.0, 0.0]},
        )
        return cls(
            exposure_offset=float(d.get("exposure_offset", 0.0)),
            vignetting_alpha=torch.tensor(
                d.get("vignetting_alpha", [[0.0] * 3] * 3), dtype=torch.float32
            ),
            vignetting_center=torch.tensor(
                d.get("vignetting_center", [0.0, 0.0]), dtype=torch.float32
            ),
            color_offsets={k: torch.tensor(v, dtype=torch.float32) for k, v in raw_color.items()},
            crf_tau=torch.tensor(d.get("crf_tau", [0.0] * 3), dtype=torch.float32),
            crf_eta=torch.tensor(d.get("crf_eta", [0.0] * 3), dtype=torch.float32),
            crf_xi=torch.tensor(d.get("crf_xi", [0.0] * 3), dtype=torch.float32),
            crf_gamma=torch.tensor(d.get("crf_gamma", [0.0] * 3), dtype=torch.float32),
        )

    def save(self, path: str) -> None:
        """Save parameters to a JSON file.

        Args:
            path: File path for the JSON output.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineParams":
        """Load parameters from a JSON file.

        Args:
            path: Path to a previously saved JSON file.

        Returns:
            Reconstructed PipelineParams instance.
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class PipelineResult:
    """Output of the ISP pipeline, with optional intermediate step images.

    Attributes:
        final: The final processed image tensor.
        intermediates: Optional dict mapping module class names to intermediate output tensors.
            Keys are set when ``return_intermediates=True`` is passed to the pipeline.
        params_used: Optional snapshot of parameters used during this forward pass.
    """

    final: Tensor
    intermediates: dict[str, Tensor] | None = None
    params_used: PipelineParams | None = None

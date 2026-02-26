"""
Named preset system for camera/film simulation.

Presets are collections of :class:`~easyppisp.params.PipelineParams` that
approximate the look of specific cameras or film stocks.

CRF preset values are expressed as **raw** (unconstrained) parameters because
PipelineParams stores them in raw form.  The actual physical values after
applying softplus/sigmoid constraints are shown in comments.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .params import PipelineParams
from .modules import (
    ISPPipeline,
    ExposureOffset,
    Vignetting,
    ColorCorrection,
    CameraResponseFunction,
)


# ---------------------------------------------------------------------------
# Built-in Preset Database
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Built-in Preset Database
# ---------------------------------------------------------------------------

_BUILTIN_PRESETS: dict[str, PipelineParams] = {
    "default": PipelineParams(),   # pure identity

    "kodak_portra_400": PipelineParams.from_constrained(
        vignetting_alpha=torch.tensor([
            [-0.15,  0.02, -0.001],  # Red channel
            [-0.15,  0.02, -0.001],  # Green channel
            [-0.18,  0.03, -0.001],  # Blue channel (slightly stronger)
        ]),
        color_offsets={
            "R": torch.tensor([ 0.02,  0.00]),
            "G": torch.tensor([ 0.00,  0.00]),
            "B": torch.tensor([ 0.00, -0.02]),
            "W": torch.tensor([ 0.01,  0.01]),
        },
        crf_tau_phys=[0.9, 0.95, 0.85],
        crf_eta_phys=[1.1, 1.0,  1.2],
        crf_xi_phys=[0.45, 0.48, 0.42],
        crf_gamma_phys=[0.78, 0.80, 0.75],
    ),

    "fuji_velvia_50": PipelineParams.from_constrained(
        vignetting_alpha=torch.tensor([
            [-0.20,  0.03, -0.002],
            [-0.20,  0.03, -0.002],
            [-0.20,  0.03, -0.002],
        ]),
        color_offsets={
            "R": torch.tensor([ 0.03,  0.01]),
            "G": torch.tensor([-0.01,  0.02]),
            "B": torch.tensor([ 0.00, -0.03]),
            "W": torch.tensor([ 0.00,  0.00]),
        },
        crf_tau_phys=[1.3, 1.2, 1.4],
        crf_eta_phys=[0.9, 0.85, 0.95],
        crf_xi_phys=[0.52, 0.50, 0.55],
        crf_gamma_phys=[0.9, 0.95, 0.88],
    ),

    "identity": PipelineParams.from_constrained(
        crf_tau_phys=[1.0, 1.0, 1.0],
        crf_eta_phys=[1.0, 1.0, 1.0],
        crf_xi_phys=[0.5, 0.5, 0.5],
        crf_gamma_phys=[1.0, 1.0, 1.0],
    ),
}


# ---------------------------------------------------------------------------
# FilmPreset API
# ---------------------------------------------------------------------------


class FilmPreset:
    """Named preset management for camera / film simulation looks.

    Example:
        >>> FilmPreset.list_presets()
        ['default', 'kodak_portra_400', 'fuji_velvia_50', 'identity']
        >>> pipeline = FilmPreset.load("kodak_portra_400")
        >>> result = pipeline(image)
    """

    @classmethod
    def list_presets(cls) -> list[str]:
        """Return all available preset names.

        Returns:
            Sorted list of preset name strings.
        """
        return sorted(_BUILTIN_PRESETS.keys())

    @classmethod
    def load(cls, name: str) -> ISPPipeline:
        """Load a named preset as a configured :class:`~easyppisp.modules.ISPPipeline`.

        Args:
            name: Preset name. See :meth:`list_presets` for available options.

        Returns:
            :class:`~easyppisp.modules.ISPPipeline` with all parameters initialized
            from the preset.

        Raises:
            KeyError: If *name* is not a recognized preset.
        """
        if name not in _BUILTIN_PRESETS:
            raise KeyError(
                f"Preset '{name}' not found. "
                f"Available presets: {cls.list_presets()}"
            )
        return ISPPipeline.from_params(_BUILTIN_PRESETS[name])

    @classmethod
    def save_params(cls, name: str, params: PipelineParams, path: str | Path) -> None:
        """Save a custom preset to a JSON file.

        Args:
            name: Preset name (stored as metadata in the file).
            params: Parameters to serialize.
            path: Output file path.
        """
        data = {"name": name, **params.to_dict()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str | Path) -> ISPPipeline:
        """Load a custom preset from a JSON file.

        Args:
            path: Path to a JSON file previously saved with :meth:`save_params`.

        Returns:
            Configured :class:`~easyppisp.modules.ISPPipeline`.
        """
        with open(path) as f:
            data = json.load(f)
        data.pop("name", None)
        params = PipelineParams.from_dict(data)
        return ISPPipeline.from_params(params)


def load_preset(name: str) -> ISPPipeline:
    """Convenience alias for :meth:`FilmPreset.load`.

    Args:
        name: Preset name.

    Returns:
        Configured :class:`~easyppisp.modules.ISPPipeline`.
    """
    return FilmPreset.load(name)

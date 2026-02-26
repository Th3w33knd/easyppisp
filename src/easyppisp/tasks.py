"""
High-level task workflows combining modules and functions.

Provides ready-to-use workflows for common developer use cases:
  - :class:`CameraSimulator`      — apply a named camera preset to images
  - :class:`PhysicalAugmentation` — thread-safe physically-plausible data augmentation
  - :class:`CameraMatchPair`      — optimize ISP params to match one camera's look to another

All task classes are designed to be safe inside PyTorch DataLoader workers
(``num_workers > 0``) by never mutating shared module state during ``__call__``.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .modules import ISPPipeline
from .params import PipelineParams
from .functional import apply_exposure, apply_vignetting, apply_color_correction, apply_crf
from .validation import check_linear_radiance

logger = logging.getLogger("easyppisp")


# ---------------------------------------------------------------------------
# CameraSimulator
# ---------------------------------------------------------------------------


class CameraSimulator:
    """Apply a complete camera simulation preset to images.

    Loads a named preset (or accepts a custom :class:`~easyppisp.modules.ISPPipeline`)
    and applies it in inference mode.

    Args:
        preset: Name of a built-in preset (see :func:`~easyppisp.presets.FilmPreset.list_presets`).
            Ignored if *pipeline* is provided.
        pipeline: Custom pre-built :class:`~easyppisp.modules.ISPPipeline`.
            If provided, *preset* is ignored.
        device: Target device for the pipeline.

    Example:
        >>> cam = CameraSimulator("kodak_portra_400")
        >>> cam.set_exposure(-0.5)
        >>> result = cam(image)
    """

    def __init__(
        self,
        preset: str = "default",
        pipeline: ISPPipeline | None = None,
        device: str = "cpu",
    ) -> None:
        if pipeline is not None:
            self.pipeline = pipeline.to(device)
        else:
            from .presets import load_preset
            self.pipeline = load_preset(preset).to(device)
        self.pipeline.eval()

    @classmethod
    def from_preset(cls, name: str, device: str = "cpu") -> "CameraSimulator":
        """Construct from a preset name (alias for ``__init__``).

        Args:
            name: Preset name.
            device: Target device.
        """
        return cls(preset=name, device=device)

    @classmethod
    def from_params(cls, params: PipelineParams, device: str = "cpu") -> "CameraSimulator":
        """Construct from a :class:`~easyppisp.params.PipelineParams` instance.

        Args:
            params: Full pipeline configuration.
            device: Target device.
        """
        return cls(pipeline=ISPPipeline.from_params(params), device=device)

    def __call__(self, image: Tensor) -> Tensor:
        """Apply the camera simulation to an image.

        Args:
            image: Linear radiance image ``(H, W, 3)`` or ``(B, H, W, 3)``.

        Returns:
            Tone-mapped output image of the same shape.
        """
        check_linear_radiance(image)
        with torch.no_grad():
            result = self.pipeline(image)
            return result.final

    def set_exposure(self, ev: float) -> None:
        """Adjust the global exposure offset in-place.

        Args:
            ev: New exposure offset in EV stops.
        """
        for mod in self.pipeline.pipeline:
            from .modules import ExposureOffset
            if isinstance(mod, ExposureOffset):
                mod.delta_t.data.fill_(ev)
                logger.debug("CameraSimulator exposure set to %.2f EV", ev)
                return
        logger.warning("CameraSimulator: no ExposureOffset module found in pipeline.")

    def set_white_balance(self, temperature_k: float) -> None:
        """Adjust the color correction toward a given color temperature.

        Converts from color temperature (Kelvin) to approximate chromaticity offsets
        using a simplified Planckian locus mapping.

        Args:
            temperature_k: Color temperature in Kelvin. Typical range: 2000–10000 K.
                Lower = warmer (orange), Higher = cooler (blue).
        """
        # Simplified reciprocal mega-kelvin (MRD) conversion.
        # This is a first-order approximation, not a calibrated sensor model.
        mrd = 1e6 / max(temperature_k, 100.0)
        # Blue channel shifts cooler with higher temperature; Red shifts warmer.
        r_shift = float(torch.tensor(-(mrd - 200.0) / 4000.0).clamp(-0.1, 0.1))
        b_shift = float(torch.tensor( (mrd - 200.0) / 4000.0).clamp(-0.1, 0.1))

        from .modules import ColorCorrection
        for mod in self.pipeline.pipeline:
            if isinstance(mod, ColorCorrection):
                mod.r_off.data[0] = r_shift
                mod.b_off.data[0] = b_shift
                logger.debug(
                    "White balance set to %.0f K (r_shift=%.4f, b_shift=%.4f)",
                    temperature_k, r_shift, b_shift,
                )
                return
        logger.warning("CameraSimulator: no ColorCorrection module found in pipeline.")


# ---------------------------------------------------------------------------
# PhysicalAugmentation
# ---------------------------------------------------------------------------


class PhysicalAugmentation:
    """Physically-plausible random augmentation for ML training data.

    Thread-safe implementation: random parameters are generated *locally* on
    each call without mutating any shared module state, making this safe to
    use with PyTorch ``DataLoader(num_workers > 0)``.

    Args:
        exposure_range: ``(min, max)`` EV range for random exposure. Default (-2, +2).
        vignetting_range: ``(min, max)`` for random α₁ coefficient magnitude.
            Set to ``(0.0, 0.0)`` to disable vignetting augmentation.
        color_jitter: Maximum per-channel chromaticity offset magnitude.
            Set to ``0.0`` to disable color jitter.

    Example:
        >>> aug = PhysicalAugmentation(exposure_range=(-1.5, 1.5), vignetting_range=(0, 0.2))
        >>> for batch in dataloader:
        ...     augmented = aug(batch)
    """

    def __init__(
        self,
        exposure_range: tuple[float, float] = (-2.0, 2.0),
        vignetting_range: tuple[float, float] = (0.0, 0.3),
        color_jitter: float = 0.02,
        crf_jitter: float = 0.05,
    ) -> None:
        self.exp_min, self.exp_max = exposure_range
        self.vig_min, self.vig_max = vignetting_range
        self.color_jitter = color_jitter
        self.crf_jitter = crf_jitter

    def __call__(self, image: Tensor) -> Tensor:
        """Apply randomly sampled physical augmentations to *image*.

        Args:
            image: Linear radiance image ``(H, W, 3)`` or ``(B, H, W, 3)``.

        Returns:
            Augmented image of the same shape, in [0, 1].
        """
        dev, dtyp = image.device, image.dtype

        # -- Stage 1: Random exposure --
        rand_ev = (
            torch.empty(1, device=dev, dtype=dtyp)
            .uniform_(self.exp_min, self.exp_max)
        )
        x = apply_exposure(image, rand_ev.squeeze())
        logger.debug("PhysicalAugmentation: exposure %.3f EV", rand_ev.item())

        # -- Stage 2: Random vignetting (if range non-zero) --
        if self.vig_max > 0.0:
            alpha_scale = (
                torch.empty(1, device=dev, dtype=dtyp)
                .uniform_(self.vig_min, self.vig_max)
                .item()
            )
            alpha = torch.zeros(3, 3, device=dev, dtype=dtyp)
            alpha[:, 0] = -alpha_scale   # α₁ (r²): falloff coefficient (negative = darker at edges)
            center = torch.zeros(2, device=dev, dtype=dtyp)
            x = apply_vignetting(x, alpha, center)
            logger.debug("PhysicalAugmentation: vignetting alpha=%.4f", alpha_scale)

        # -- Stage 3: Random color jitter --
        if self.color_jitter > 0.0:
            jitter = lambda: (
                torch.empty(2, device=dev, dtype=dtyp)
                .uniform_(-self.color_jitter, self.color_jitter)
            )
            color_offsets = {
                "R": jitter(),
                "G": jitter(),
                "B": jitter(),
                "W": jitter(),
            }
            x = apply_color_correction(x, color_offsets)

        # -- Stage 4: Random CRF jitter --
        if self.crf_jitter > 0.0:
            crf_jitter = lambda: (
                torch.empty(3, device=dev, dtype=dtyp)
                .uniform_(-self.crf_jitter, self.crf_jitter)
            )
            x = apply_crf(x, crf_jitter(), crf_jitter(), crf_jitter(), crf_jitter())

        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# CameraMatchPair
# ---------------------------------------------------------------------------


class CameraMatchPair:
    """Optimize ISP parameters to match the visual style of one camera to another.

    Minimizes the L2 distance between the ISP-processed *source* images and the
    *target* images.  After fitting, :meth:`transform` applies the learned mapping
    to new source images.

    Args:
        device: Device for optimization.
        lr: Learning rate for Adam optimizer.

    Example:
        >>> matcher = CameraMatchPair(device="cuda")
        >>> matcher.fit(source_images, target_images, num_steps=500)
        >>> matched = matcher.transform(new_source_image)
        >>> matcher.save_params("match.json")
    """

    def __init__(self, device: str = "cpu", lr: float = 5e-3) -> None:
        self.device = device
        self.lr = lr
        self.pipeline: ISPPipeline | None = None

    def fit(
        self,
        source_images: list[Tensor],
        target_images: list[Tensor],
        num_steps: int = 500,
        verbose: bool = True,
    ) -> "CameraMatchPair":
        """Optimize ISP parameters to map source images to look like target images.

        Args:
            source_images: List of source camera images (linear, ``(H, W, 3)``).
            target_images: List of corresponding target camera images (linear, ``(H, W, 3)``).
            num_steps: Number of Adam optimization steps.
            verbose: Print loss every 100 steps.

        Returns:
            Self (for method chaining).
        """
        assert len(source_images) == len(target_images), (
            f"source and target must have the same number of images, "
            f"got {len(source_images)} vs {len(target_images)}"
        )

        self.pipeline = ISPPipeline().to(self.device)
        self.pipeline.train()

        optimizer = torch.optim.Adam(self.pipeline.parameters(), lr=self.lr)

        src_tensors = [img.to(self.device) for img in source_images]
        tgt_tensors = [img.to(self.device) for img in target_images]

        for step in range(num_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            for src, tgt in zip(src_tensors, tgt_tensors):
                pred = self.pipeline(src).final
                total_loss = total_loss + torch.nn.functional.mse_loss(pred, tgt)
            total_loss = total_loss / len(src_tensors)
            total_loss.backward()
            optimizer.step()

            if verbose and (step + 1) % 100 == 0:
                logger.info("CameraMatchPair step %d/%d  loss=%.6f", step + 1, num_steps, total_loss.item())
                print(f"  step {step+1:4d}/{num_steps}  loss={total_loss.item():.6f}")

        self.pipeline.eval()
        return self

    def transform(self, image: Tensor) -> Tensor:
        """Apply the learned mapping to a new source image.

        Args:
            image: New source image in linear radiance ``(H, W, 3)``.

        Returns:
            Mapped image in the target camera's visual style.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Call CameraMatchPair.fit() before transform().")
        with torch.no_grad():
            return self.pipeline(image.to(self.device)).final

    def save_params(self, path: str) -> None:
        """Save the fitted pipeline parameters to a JSON file.

        The file is written in :class:`~easyppisp.params.PipelineParams` format
        so it can be reloaded with ``PipelineParams.load(path)`` or
        ``FilmPreset.load_from_file(path)``.

        Args:
            path: Output file path.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Call CameraMatchPair.fit() before save_params().")

        # Build a PipelineParams from the current module state so the serialized
        # format is compatible with PipelineParams.load() and FilmPreset.load_from_file().
        from .params import PipelineParams
        from .modules import ExposureOffset, Vignetting, ColorCorrection, CameraResponseFunction

        p = PipelineParams()
        for mod in self.pipeline.pipeline:
            if isinstance(mod, ExposureOffset):
                p.exposure_offset = mod.delta_t.item()
            elif isinstance(mod, Vignetting):
                p.vignetting_alpha = mod.alpha.detach().cpu()
                p.vignetting_center = mod.center.detach().cpu()
            elif isinstance(mod, ColorCorrection):
                p.color_offsets = {
                    "B": mod.b_off.detach().cpu(),
                    "R": mod.r_off.detach().cpu(),
                    "G": mod.g_off.detach().cpu(),
                    "W": mod.w_off.detach().cpu(),
                }
            elif isinstance(mod, CameraResponseFunction):
                # Store raw (unconstrained) values — matches PipelineParams fields
                p.crf_tau   = mod.tau.detach().cpu()
                p.crf_eta   = mod.eta.detach().cpu()
                p.crf_xi    = mod.xi.detach().cpu()
                p.crf_gamma = mod.gamma.detach().cpu()

        p.save(path)
        logger.info("CameraMatchPair params saved to %s", path)

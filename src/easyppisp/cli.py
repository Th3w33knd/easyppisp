"""
Command-line interface for easyppisp.

Usage:
    easyppisp apply [--exposure FLOAT] [--preset NAME] INPUT OUTPUT
    easyppisp presets

Requires Pillow (``pip install easyppisp[dev]``).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        print(
            "Error: The easyppisp CLI requires Pillow.\n"
            "Install it with: pip install easyppisp[dev]",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_apply(args: argparse.Namespace) -> None:
    """Apply ISP adjustments to an image file."""
    _require_pillow()

    from PIL import Image
    from .utils import from_pil, to_pil
    from .presets import load_preset
    from .functional import apply_exposure
    from .modules import ExposureOffset

    img_pil = Image.open(args.input)
    linear = from_pil(img_pil)

    if args.preset and args.preset != "default":
        pipeline = load_preset(args.preset)
        # Override exposure if explicitly requested
        if args.exposure != 0.0:
            for mod in pipeline.pipeline:
                if isinstance(mod, ExposureOffset):
                    mod.delta_t.data.fill_(args.exposure)
        import torch
        with torch.no_grad():
            result = pipeline(linear).final
    else:
        result = apply_exposure(linear, args.exposure)

    out_pil = to_pil(result.clamp(0.0, 1.0))
    out_pil.save(str(args.output))
    print(f"Saved: {args.output}")


def cmd_augment(args: argparse.Namespace) -> None:
    """Apply random physical augmentations to an image."""
    _require_pillow()
    from PIL import Image
    from .utils import from_pil, to_pil
    from .tasks import PhysicalAugmentation

    img_pil = Image.open(args.input)
    linear = from_pil(img_pil)

    aug = PhysicalAugmentation(
        exposure_range=(args.exposure_min, args.exposure_max),
        vignetting_range=(args.vignetting_min, args.vignetting_max),
        color_jitter=args.color_jitter,
        crf_jitter=args.crf_jitter,
    )

    for i in range(args.n):
        result = aug(linear)
        out_pil = to_pil(result.clamp(0.0, 1.0))
        
        if args.n > 1:
             out_path = args.output.parent / f"{args.output.stem}_{i:03d}{args.output.suffix}"
        else:
             out_path = args.output
             
        out_pil.save(str(out_path))
        print(f"Saved: {out_path}")


def cmd_match(args: argparse.Namespace) -> None:
    """Optimize ISP parameters to match source images to target style."""
    _require_pillow()
    import torch
    from PIL import Image
    from .utils import from_pil
    from .tasks import CameraMatchPair

    def _load_dir(d: Path) -> list[torch.Tensor]:
        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        files = sorted([f for f in d.iterdir() if f.suffix.lower() in exts])
        return [from_pil(Image.open(f)) for f in files]

    print(f"Loading source images from {args.source_dir}...")
    src_images = _load_dir(args.source_dir)
    print(f"Loading target images from {args.target_dir}...")
    tgt_images = _load_dir(args.target_dir)

    if len(src_images) != len(tgt_images):
        print(f"Error: Found {len(src_images)} source images but {len(tgt_images)} target images.")
        sys.exit(1)

    matcher = CameraMatchPair(lr=args.lr)
    matcher.fit(src_images, tgt_images, num_steps=args.num_steps)

    if args.save:
        matcher.save_params(args.save)
        print(f"Optimization complete. Parameters saved to {args.save}")
    else:
        params = matcher.pipeline.get_params_dict()
        print("\nOptimization complete. Learned parameters:")
        print(params)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="easyppisp",
        description="EasyPPISP — Physically-Plausible ISP toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- apply --
    apply_p = sub.add_parser("apply", help="Apply ISP effects to an image")
    apply_p.add_argument("input",  type=Path, help="Input image path")
    apply_p.add_argument("output", type=Path, help="Output image path")
    apply_p.add_argument(
        "--exposure", type=float, default=0.0,
        help="Exposure offset in EV stops (default: 0.0)",
    )
    apply_p.add_argument(
        "--preset", type=str, default="default",
        help="Named camera/film preset to apply (default: 'default')",
    )

    # -- augment --
    aug_p = sub.add_parser("augment", help="Apply physical augmentations")
    aug_p.add_argument("input", type=Path)
    aug_p.add_argument("output", type=Path)
    aug_p.add_argument("-n", type=int, default=1, help="Number of variations")
    aug_p.add_argument("--exposure-min", type=float, default=-1.0)
    aug_p.add_argument("--exposure-max", type=float, default=1.0)
    aug_p.add_argument("--vignetting-min", type=float, default=0.0)
    aug_p.add_argument("--vignetting-max", type=float, default=0.2)
    aug_p.add_argument("--color-jitter", type=float, default=0.02)
    aug_p.add_argument("--crf-jitter", type=float, default=0.05)

    # -- match --
    match_p = sub.add_parser("match", help="Match visual style of two cameras")
    match_p.add_argument("--source-dir", type=Path, required=True)
    match_p.add_argument("--target-dir", type=Path, required=True)
    match_p.add_argument("--num-steps", type=int, default=500)
    match_p.add_argument("--lr", type=float, default=5e-3)
    match_p.add_argument("--save", type=str, help="Path to save learned params (.json)")

    # -- presets --
    sub.add_parser("presets", help="List available built-in presets")

    args = parser.parse_args()

    if args.command == "apply":
        cmd_apply(args)
    elif args.command == "augment":
        cmd_augment(args)
    elif args.command == "match":
        cmd_match(args)
    elif args.command == "presets":
        cmd_presets(args)


if __name__ == "__main__":
    main()

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


def cmd_presets(_args: argparse.Namespace) -> None:
    """List available built-in presets."""
    from .presets import FilmPreset
    names = FilmPreset.list_presets()
    print("Available presets:")
    for name in names:
        print(f"  {name}")


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

    # -- presets --
    sub.add_parser("presets", help="List available built-in presets")

    args = parser.parse_args()

    if args.command == "apply":
        cmd_apply(args)
    elif args.command == "presets":
        cmd_presets(args)


if __name__ == "__main__":
    main()

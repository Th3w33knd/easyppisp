# EasyPPISP

A developer-friendly Python wrapper for NVIDIA's [Physically-Plausible ISP (PPISP)](https://arxiv.org/abs/2601.18336) framework. Simulates real camera physics — Exposure, Vignetting, White Balance, and non-linear Sensor Response — with clean, composable, differentiable PyTorch modules.

## Installation

```bash
pip install easyppisp           # Core (requires PyTorch)
pip install easyppisp[dev]      # + Pillow + pytest
```

All pipeline stages are implemented in pure PyTorch and are fully differentiable on both CPU and CUDA. The NVIDIA `ppisp` CUDA kernel can be used directly by callers for production training where maximum throughput is needed.

## Quickstart (≤ 5 lines)

```python
import easyppisp
from easyppisp.utils import load_image, save_image

image = load_image("photo.jpg")               # linear float32 HWC tensor
bright = easyppisp.apply(image, exposure=1.5) # +1.5 stops of exposure
save_image(bright, "bright.jpg")
```

## Architecture

```
Raw Radiance → Exposure → Vignetting → Color Correction → CRF → Final Image
               (Eq. 3)    (Eq. 4–5)    (Eq. 6–12)       (Eq. 13–16)
```

### Functional API — stateless, differentiable

```python
from easyppisp.functional import apply_exposure, apply_vignetting, apply_color_correction, apply_crf

x = apply_exposure(image, delta_t=1.0)        # +1 EV
x = apply_vignetting(x, alpha, center)        # radial falloff
x = apply_color_correction(x, color_offsets) # white balance
x = apply_crf(x, tau_raw, eta_raw, xi_raw, gamma_raw)  # tone map
```

### Module API — composable `nn.Module` wrappers

```python
from easyppisp import ISPPipeline, ExposureOffset, Vignetting, CameraResponseFunction

# Custom pipeline — only the stages you need, in any order
pipeline = ISPPipeline([
    ExposureOffset(delta_t=-0.5),
    Vignetting(alpha=torch.tensor([[-0.2, 0.0, 0.0]] * 3)),
    CameraResponseFunction(),
])

result = pipeline(image, return_intermediates=True)
print(result.intermediates.keys())
# dict_keys(['ExposureOffset', 'Vignetting', 'CameraResponseFunction'])
```

### Task API — high-level workflows

```python
from easyppisp import CameraSimulator, PhysicalAugmentation, CameraMatchPair

# Named camera/film presets
cam = CameraSimulator("kodak_portra_400")
cam.set_exposure(-0.5)
result = cam(image)

# Thread-safe data augmentation for DataLoaders
aug = PhysicalAugmentation(exposure_range=(-2.0, 2.0), vignetting_range=(0, 0.3))
augmented = aug(batch)   # safe with num_workers > 0

# Camera-to-camera matching
matcher = CameraMatchPair()
matcher.fit(source_images, target_images, num_steps=500)
matched = matcher.transform(new_image)
matcher.save_params("sony_to_iphone.json")
```

### Presets

```python
from easyppisp import FilmPreset

FilmPreset.list_presets()
# ['default', 'fuji_velvia_50', 'identity', 'kodak_portra_400']

pipeline = FilmPreset.load("fuji_velvia_50")
# Save a custom preset
FilmPreset.save_params("my_camera", params, "my_camera.json")
pipeline = FilmPreset.load_from_file("my_camera.json")
```

### CLI

```bash
easyppisp apply --exposure +1.5 photo.jpg bright.jpg
easyppisp apply --preset kodak_portra_400 photo.jpg film.jpg
easyppisp presets
```

## Key Design Decisions

| Issue in v4 draft | Fix in easyppisp |
|---|---|
| Import errors — `Vignetting`, `ColorCorrection`, `CameraResponseFunction` missing | All four stages are independent `nn.Module` subclasses |
| Phantom import `from ._internal.crf_curves import ...` | CRF implemented inline in `functional.py` with constraints |
| Missing `torch` import in `__init__.py` | Fixed |
| Color ordering mismatch (R,G,B,W vs required B,R,G,W) | `apply_color_correction` enforces B→R→G→W ordering internally |
| Unconstrained CRF parameters → NaN / non-monotonic | `softplus` + `sigmoid` applied inside `apply_crf` |
| `PhysicalAugmentation` mutates shared state (thread-unsafe) | Fully stateless — parameters generated per-call |
| Monolithic `ISPPipeline` (not composable) | `ISPPipeline` accepts any `nn.ModuleList` sequence |
| `check_linear_radiance` false positives | Heuristic uses `max > 10.0` threshold only |
| Default pipeline not identity (gamma≈0.8) | `PipelineParams` CRF defaults pre-inverted to produce tau=eta=gamma=1, xi=0.5 |
| `apply()` silently swallows `**kwargs` | Replaced with explicit named parameters; all forwarded to `apply_pipeline` |
| `get_params_dict()` vs `PipelineParams` key mismatch | CRF dict now includes both `_raw` and `_phys` keys; raw keys match `PipelineParams` |
| `CameraMatchPair.save_params()` incompatible format | Writes `PipelineParams`-compatible flat JSON; loadable by `PipelineParams.load()` |
| Dead CUDA import claiming delegation that never happened | Removed; README corrected to accurately describe pure-PyTorch implementation |
| `from_pil()` always applies inverse-gamma (double-linearizes HDR TIFFs) | Added `linearize=True` parameter; set `False` for already-linear sources |

## Testing

```bash
pip install easyppisp[dev]
pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Built on PPISP ([NVIDIA, 2025](https://arxiv.org/abs/2601.18336)).

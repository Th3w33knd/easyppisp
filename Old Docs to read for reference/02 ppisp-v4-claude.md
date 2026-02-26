# Critical Analysis of `easyppisp` Codebase

---

## 1. Structural Issues

### 1.1 - Package Layout Ambiguity: `_internal/` Directory

`functional.py` imports:
```python
from ._internal.color_homography import build_homography, apply_homography
```

The repository file listing shows `color_homography.py` at the package root - no `_internal/` directory, no `_internal/__init__.py`. If the listing is accurate, this is a `ModuleNotFoundError` that crashes every import. If the listing is a flattened view and the actual file lives at `_internal/color_homography.py`, then the missing `_internal/__init__.py` (not present among the 18 files) relies on Python 3.3+ namespace package mechanics, which works but is non-standard for a regular package and omitted from the project structure documentation.

The README, tests, and every example assume `import easyppisp` succeeds. This is the single point of failure for the entire library. Either the listing is incomplete (an `_internal/__init__.py` exists but wasn't captured) or the import path is wrong and should be `from .color_homography import ...`.

### 1.2 - CUDA Backend Imported But Never Used

```python
# functional.py
try:
    import ppisp as _ppisp_backend
    HAS_PPISP_CUDA = True
except ImportError:
    HAS_PPISP_CUDA = False
```

`HAS_PPISP_CUDA` is set but never read. `_ppisp_backend` is imported but never called. No code path delegates to the CUDA kernel. This is dead code.

The README states: *"The PPISP CUDA kernel (ppisp) is used automatically when installed. The library falls back to pure-PyTorch implementations when it is not available."*

This is false. The CUDA kernel is never used regardless of installation status. To the library's credit, the previous versions' CUDA delegation code had critical bugs (shape mismatches, parameter domain inconsistencies). Removing the broken delegation and keeping only the pure-PyTorch path is functionally safer, but the documentation claim should be corrected.

---

## 2. Identity Invariant

### 2.1 - Default Pipeline Is Not Identity

The prompt requires: *"Default parameters must produce identity (no-op) transforms so the pipeline is opt-in per module."*

`PipelineParams` defaults all CRF raw parameters to zero. After constraints:

```python
tau   = 0.3 + softplus(0) ≈ 0.993
eta   = 0.3 + softplus(0) ≈ 0.993
xi    = sigmoid(0)         = 0.5
gamma = 0.1 + softplus(0) ≈ 0.793
```

Tracing a pixel value of 0.7 through the default CRF:
- S-curve with tau≈eta≈1: approximately passes through → y ≈ 0.699
- Gamma: `0.699^0.793 ≈ 0.762`

So `ISPPipeline()` transforms `0.7 → 0.762`. The default pipeline is not a no-op.

The docstring is transparent about this:
```python
# Default raw=0: tau=0.3+ln2≈1.0, eta≈1.0, xi=0.5, gamma≈0.8
```

And the preset system provides a separate `"identity"` preset with correctly inverted raw values:
```python
_id_tau, _id_eta, _id_xi, _id_gamma = _make_crf_raw(
    [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]
)
```

This is a deliberate design choice: raw=0 centers the unconstrained parameter space for optimization stability, at the cost of non-identity defaults. The tradeoff is reasonable for an optimization-focused library, but no test verifies the invariant either way. The test `test_load_default_is_identity` has a misleading docstring ("identity params") but correctly only checks output range.

### 2.2 - `apply_pipeline` Is Identity, `ISPPipeline` Is Not

The functional `apply_pipeline()` skips stages when parameters are `None`:
```python
if all(t is not None for t in (crf_tau_raw, ...)):
    x = apply_crf(x, ...)
```

So `apply_pipeline(image)` with all defaults returns `image` unchanged. But `ISPPipeline()` always includes a `CameraResponseFunction()` that applies the non-identity CRF. Two paths to "defaults" produce different results, which could confuse users switching between functional and module APIs.

---

## 3. API Design Issues

### 3.1 - `apply()` Silently Swallows Keyword Arguments

```python
def apply(image, exposure=0.0, preset=None, **kwargs) -> Tensor:
    """...
    **kwargs: Additional keyword arguments forwarded to :func:`apply_pipeline`.
    """
    if preset is not None:
        cam = CameraSimulator(preset=preset)
        cam.set_exposure(exposure)
        return cam(image)
    return apply_exposure(image, exposure)
```

The `**kwargs` are accepted but never forwarded anywhere. The docstring claims they go to `apply_pipeline`, but `apply_pipeline` is never called. A user who writes:
```python
easyppisp.apply(img, exposure=1.0, vignetting_alpha=my_alpha)
```
gets the vignetting silently ignored. This violates the fail-fast principle.

### 3.2 - `get_params_dict()` and `PipelineParams` Use Incompatible Formats

`CameraResponseFunction.get_params_dict()` returns **constrained** CRF values:
```python
tau = (0.3 + F.softplus(self.tau)).detach().tolist()
return {"crf_tau": tau, ...}
```

`PipelineParams.to_dict()` stores **raw** (unconstrained) values:
```python
return {"crf_tau": self.crf_tau.tolist(), ...}  # raw values
```

Both use the same key name `"crf_tau"` but in different value domains. A user who calls:
```python
params_dict = pipeline.get_params_dict()["CameraResponseFunction"]
PipelineParams.from_dict(params_dict)  # WRONG: treats constrained as raw
```
gets silently incorrect parameters. The constrained tau≈0.993 would be re-constrained to `0.3 + softplus(0.993) ≈ 1.593`.

Additionally, `CameraMatchPair.save_params()` writes the nested `get_params_dict()` format:
```json
{"ExposureOffset": {"exposure_offset_ev": 0.5}, "CameraResponseFunction": {"crf_tau": [0.993, ...]}}
```

This cannot be loaded by `PipelineParams.load()`, which expects the flat format:
```json
{"exposure_offset": 0.5, "crf_tau": [0.014, ...]}
```

The key names differ too: `exposure_offset_ev` vs `exposure_offset`. A trained camera match cannot be round-tripped through the serialization system.

### 3.3 - `from_pil()` Always Linearizes With No Escape Hatch

```python
def from_pil(image, device="cpu") -> Tensor:
    arr = torch.from_numpy(np.asarray(image.convert("RGB"))).to(device)
    return srgb_to_linear(from_uint8(arr))
```

A user loading a linear-light 16-bit TIFF or EXR via PIL gets double-linearized. V4 added a `linearize: bool = True` parameter. This version doesn't provide one. The docstring correctly describes the assumption, but there's no workaround short of manually calling `from_uint8` without `srgb_to_linear`.

### 3.4 - `set_white_balance` Is a Rough Approximation

```python
def set_white_balance(self, temperature_k: float) -> None:
    mrd = 1e6 / max(temperature_k, 100.0)
    r_shift = float(torch.tensor(-(mrd - 200.0) / 4000.0).clamp(-0.1, 0.1))
    b_shift = float(torch.tensor( (mrd - 200.0) / 4000.0).clamp(-0.1, 0.1))
    mod.r_off.data[0] = r_shift
    mod.b_off.data[0] = b_shift
```

This is documented as "a first-order approximation, not a calibrated sensor model," which is honest. But for a library whose value proposition is *physically plausible* simulation:
1. It only modifies red and blue - green is untouched
2. It modifies only the first chromaticity coordinate `Δr`, not `Δg`
3. The linear reciprocal mega-kelvin mapping is a rough approximation of the nonlinear Planckian locus

The documentation is adequate, but the function name suggests precision it doesn't deliver.

---

## 4. Missing Deliverables

### 4.1 - Specified Components Not Implemented

| Component | Status | Impact |
|---|---|---|
| `ISPController` (Eq. 17, Sec 4.5) | ❌ Not implemented | Core PPISP feature: novel-view parameter prediction |
| Regularization losses (Eq. 18-22) | ❌ Not implemented | Required for training the pipeline as designed in the paper |
| `PipelineParams.from_constrained()` | ❌ Only private `_make_crf_raw` in presets | Users must manually invert constraints for custom presets |
| CLI `augment` subcommand | ❌ Not implemented | Prompt specifies `easyppisp augment --count 10 ...` |
| CLI `match` subcommand | ❌ Not implemented | Prompt specifies `easyppisp match --source ... --target ...` |
| `_internal/crf_curves.py` | ❌ CRF is inline in functional.py | Prompt requires standalone internal module |
| `_internal/__init__.py` | ❌ Not provided | Needed for `_internal` to be a proper package |
| YAML preset loading | ❌ JSON only | Prompt specifies YAML-based preset system |
| CRF jitter in `PhysicalAugmentation` | ❌ Only exposure, vignetting, color | Missing sensor response curve augmentation |
| CUDA fast-path delegation | ❌ Dead import code | README claims it works; it doesn't |

### 4.2 - Documentation Not Provided

| Deliverable | Status |
|---|---|
| `docs/math_reference.md` | ❌ |
| `docs/quickstart.md` | ❌ |
| `docs/api_reference.md` | ❌ |
| `docs/tutorials/` | ❌ |
| `examples/` directory | ❌ |
| `pyproject.toml` | ❌ (referenced in README but not provided) |
| `LICENSE` | ❌ (referenced in README but not provided) |

---

## 5. Test Quality Assessment

### 5.1 - Coverage Summary

| Test File | Test Count | Runnable | Notes |
|---|---|---|---|
| `test_functional.py` | 28 | ✅ | Comprehensive: identity, known-value, shape, batch, gradient, warnings |
| `test_gradient_flow.py` | 6 | ✅ | All four pipeline stages + exposure gradient value check |
| `test_modules.py` | 26 | ✅ | Independent module tests, composition, ordering warnings, state_dict round-trip |
| `test_params.py` | 7 | ✅ | Serialization, JSON validity, type checks |
| `test_pipeline_integration.py` | 21 | ✅ | End-to-end, presets, augmentation, camera matching, differentiability |
| `test_validation.py` | 16 | ✅ | Shape errors, device checks, exposure warnings, exception hierarchy |
| **Total** | **104** | | |

This is a substantial and well-organized test suite. Each test class has a clear scope, fixture usage is consistent, and test names are descriptive.

### 5.2 - Notable Test Strengths

The `test_exposure_gradient_value` test is exceptional - it doesn't just verify gradients flow, it verifies the *correct numerical value*:
```python
def test_exposure_gradient_value(self):
    """∂(apply_exposure)/∂(delta_t) = image * ln(2) * 2^delta_t."""
    ...
    expected_grad = img.sum().item() * torch.log(torch.tensor(2.0)).item()
    assert abs(dt.grad.item() - expected_grad) < 1e-5
```

The color correction gradcheck properly passes `img` as an explicit argument (not a closure capture), ensuring the image-path Jacobian is validated:
```python
def _wrapper(img_t, b_t, r_t, g_t, w_t):
    return apply_color_correction(img_t, {"B": b_t, "R": r_t, "G": g_t, "W": w_t})
assert gradcheck(_wrapper, (img, b, r, g, w), ...)
```

The monotonicity test covers the CRF's key mathematical property:
```python
def test_monotone_channel_response(self, identity_crf_raw):
    xs = torch.linspace(0.01, 0.99, 100).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
    ys = apply_crf(xs, tau, eta, xi, gamma)
    diffs = ys[1:] - ys[:-1]
    assert (diffs >= -1e-5).all()
```

### 5.3 - Test Gaps

1. **No identity test for full pipeline**: No test asserts `ISPPipeline()(img) ≈ img` or `ISPPipeline.from_params(identity_preset)(img) ≈ img`. The "identity" preset is untested.

2. **Monotonicity only at default params**: The CRF monotonicity test uses `identity_crf_raw` (zeros). No test verifies monotonicity with random extreme parameters, which is the scenario where the `softplus`/`sigmoid` constraints are most critical.

3. **No CUDA testing**: The `device` fixture is parameterized with only `"cpu"`. No test is decorated with `@pytest.mark.cuda`. The CUDA path is completely untested.

4. **Dead fixture**: `img_hwc_f64` is defined in conftest but never used in any test.

5. **No test for `save_params` → `load` interoperability**: `CameraMatchPair.save_params()` and `PipelineParams.load()` use incompatible formats, but no test catches this.

6. **`apply()` kwargs not tested**: No test verifies that passing unrecognized kwargs to `easyppisp.apply()` behaves correctly (or raises).

---

## 6. Code Quality

### 6.1 - Strengths

**Docstrings**: Every public class, method, and function has a complete docstring with description, args, returns, raises, and often an example. This is the first version across all iterations that meets the prompt's documentation standard.

**Type hints**: Consistent use of `Tensor | None` union syntax with `from __future__ import annotations`. Return types on all public functions. Parameter types specified throughout.

**Error messages**: The validation module produces genuinely helpful errors:
```
"'image' must be (H, W, 3) [single image] or (B, H, W, 3) [batch].
Got shape (10,) (ndim=1). If your tensor is CHW, use `easyppisp.utils.chw_to_hwc()` first."
```

**Logging**: Proper use throughout - `logger.debug` for operational details, `logger.info` for training progress, `logger.warning` for concerning but non-fatal conditions.

**Thread safety**: `PhysicalAugmentation` generates all random parameters locally per call, safe for `DataLoader` workers.

**Presets**: The `_make_crf_raw` function correctly inverts constraint functions to convert human-readable physical values to raw space. Multiple presets demonstrate the system works.

### 6.2 - Minor Issues

**Constant tensor recreation**: `build_homography` calls `.to(device, dtype)` on module-level constant tensors (`_SRC_B`, `_S_INV`, `_COLOR_PINV_BLOCK_DIAG`) every invocation. When device/dtype differs from CPU float32 (e.g., CUDA training), this creates new tensors each call. Not a correctness issue, but unnecessary allocation in tight training loops.

**`CameraMatchPair.fit` no resolution handling**: If source and target images have different spatial dimensions, `torch.nn.functional.mse_loss` will crash. V4 addressed this with `F.interpolate`. This version doesn't.

**`PhysicalAugmentation` no CRF augmentation**: The prompt specifies simulating "what this looks like through a cheap lens." Different lens/sensor response curves (CRF variation) are important for this use case but not included in the augmentation pipeline.

**Epsilon consistency**: Three different values (`1e-7`, `1e-5`, `1e-6`) across two files. Each is locally documented, which is better than previous versions, but not centralized.

---

## 7. Summary Assessment

| Criterion | Rating | Notes |
|---|---|---|
| **Importable** | ⚠️ Conditional | Depends on `_internal/` directory existing (ambiguous from listing) |
| **Mathematical correctness** | ✅ Good | CRF constraints guarantee monotonicity; color ordering correct; vignetting coordinates consistent |
| **Identity invariant** | ❌ Violated | Default pipeline has gamma≈0.8; "identity" preset exists separately |
| **Completeness vs. prompt** | ~65% | Strong functional/module/task/test/preset coverage; missing ISPController, regularization, CUDA delegation, docs, examples |
| **Code quality** | ✅ High | Consistent docstrings, type hints, logging, error messages, test coverage |
| **Test suite** | ✅ Strong | 104 tests across 6 files; gaps in identity, CUDA, and interoperability testing |
| **Production readiness** | ⚠️ Close | Solid for CPU pure-PyTorch usage; serialization interop issues need fixing; CUDA claims are inaccurate |

This is the strongest implementation across all iterations. The test suite alone represents an order-of-magnitude improvement over prior versions. The docstring and error message quality meets professional library standards. The preset system with proper constraint inversion is well-engineered. The most impactful remaining issues are the missing ISPController (the paper's distinguishing feature), the serialization format mismatch between `get_params_dict()` and `PipelineParams`, the non-identity defaults, and the false CUDA delegation claim in the README.

---

# Critical Analysis of `easyppisp` - Complete Codebase (Parts 1+2)

---

## 1. Structure & Import Chain

### 1.1 - Package Is Importable and Properly Structured

The `_internal/__init__.py` exists. The directory layout is correct:

```
easyppisp/
  __init__.py
  _internal/
    __init__.py
    color_homography.py
```

Every import resolves. No circular dependencies. `import easyppisp` succeeds. `src/` exists but is empty; the package uses a flat layout rooted at `easyppisp/`.

### 1.2 - Missing `pyproject.toml`

Not present among the 19 files. The README says `pip install easyppisp` but the package is not installable without a build configuration. Not a code bug but a distribution blocker.

---

## 2. Test Bug: `test_get_params_dict_constrained_values` Fails

This is the most impactful bug in the codebase - a test that will fail on every run.

`CameraResponseFunction.get_params_dict()` was refactored to return **both** raw and physical values:

```python
return {
    "crf_tau_phys": tau_phys,      # physical ≈ [0.993, ...]
    "crf_tau": self.tau.detach().tolist(),  # raw = [0.0, ...]
    ...
}
```

But the test in `test_modules.py` still reads the raw key expecting physical values:

```python
def test_get_params_dict_constrained_values(self):
    mod = CameraResponseFunction()   # defaults to raw zeros
    d = mod.get_params_dict()
    for val in d["crf_tau"]:         # gets raw [0.0, 0.0, 0.0]
        assert val > 0.3             # FAILS: 0.0 is not > 0.3
    for val in d["crf_gamma"]:       # gets raw [0.0, 0.0, 0.0]
        assert val > 0.1             # FAILS: 0.0 is not > 0.1
    for val in d["crf_xi"]:          # gets raw [0.0, 0.0, 0.0]
        assert 0.0 < val < 1.0       # FAILS: 0.0 is not in (0, 1)
```

The test should reference `d["crf_tau_phys"]`, `d["crf_gamma_phys"]`, and `d["crf_xi_phys"]`. This test will fail every `pytest` run.

---

## 3. Identity Invariant: Split Personality

### 3.1 - `PipelineParams()` Is Identity; `CameraResponseFunction()` Is Not

`PipelineParams` correctly pre-inverts CRF defaults:

```python
_CRF_TAU_IDENTITY = math.log(math.expm1(0.7))   # → constrained tau = 1.0
_CRF_GAMMA_IDENTITY = math.log(math.expm1(0.9)) # → constrained gamma = 1.0
```

But `CameraResponseFunction()` defaults to zeros:

```python
class CameraResponseFunction(nn.Module):
    def __init__(self, tau=None, ...):
        zeros = torch.zeros(3)
        self.tau = nn.Parameter((tau if tau is not None else zeros).clone().float())
```

This creates two distinct "default" behaviors:

| Construction | CRF Raw | Gamma (Physical) | Identity? |
|---|---|---|---|
| `ISPPipeline()` | 0.0 | ≈ 0.793 | ❌ |
| `ISPPipeline.from_params(PipelineParams())` | ≈ 0.014 | 1.0 | ✅ |
| `CameraResponseFunction()` | 0.0 | ≈ 0.793 | ❌ |
| `CameraResponseFunction.from_params(PipelineParams())` | ≈ 0.014 | 1.0 | ✅ |

The `params.py` docstring incorrectly claims: *"This ensures PipelineParams() / ISPPipeline() is a true no-op."* `ISPPipeline()` is not a no-op because it creates `CameraResponseFunction()` with zero defaults, not `CameraResponseFunction.from_params(PipelineParams())`.

### 3.2 - `from_dict({})` Falls Back to Non-Identity

```python
crf_tau=torch.tensor(d.get("crf_tau", [0.0] * 3), ...)
```

The fallback `[0.0] * 3` differs from `PipelineParams()`'s `_CRF_TAU_IDENTITY ≈ 0.014`. A JSON file missing CRF keys would produce non-identity behavior on load. In practice, `PipelineParams.save()` always writes CRF keys, so proper round-trips are fine. But manually crafted or partial JSON files hit this inconsistency.

### 3.3 - Identity Tests Are Correct

The `TestIdentityInvariant` class correctly tests `ISPPipeline.from_params(PipelineParams())` (not bare `ISPPipeline()`), and the "identity" preset. These tests should pass. The naming is precise:

```python
def test_default_params_pipeline_is_identity(self, img_hwc):
    pipeline = ISPPipeline.from_params(PipelineParams())
    out = pipeline(img_hwc).final
    assert torch.allclose(out, img_hwc, atol=1e-4)
```

---

## 4. Serialization: Properly Fixed

### 4.1 - `CameraMatchPair.save_params()` Is Now Compatible

The implementation builds a proper `PipelineParams` from module state:

```python
def save_params(self, path: str) -> None:
    p = PipelineParams()
    for mod in self.pipeline.pipeline:
        if isinstance(mod, ExposureOffset):
            p.exposure_offset = mod.delta_t.item()
        elif isinstance(mod, CameraResponseFunction):
            p.crf_tau = mod.tau.detach().cpu()  # raw values
            ...
    p.save(path)
```

This produces flat JSON loadable by `PipelineParams.load()`. The `TestSerializationInterop.test_camera_match_save_loadable_by_pipeline_params` test verifies this.

### 4.2 - `get_params_dict()` Format Mismatch Still Exists (But Is Documented)

`ISPPipeline.get_params_dict()` returns a nested dict (`{"ExposureOffset": {...}, ...}`) that is NOT `PipelineParams`-compatible. But `save_params()` now bypasses `get_params_dict()` entirely, building `PipelineParams` directly. The `get_params_dict()` method is explicitly for *inspection/logging*, not serialization. This is a reasonable separation.

---

## 5. API: Well-Designed

### 5.1 - `apply()` Properly Forwards Parameters

```python
def apply(image, exposure=0.0, preset=None,
          vignetting_alpha=None, vignetting_center=None,
          color_offsets=None,
          crf_tau_raw=None, crf_eta_raw=None, crf_xi_raw=None, crf_gamma_raw=None):
```

Explicit named parameters, no `**kwargs`. All forwarded to `apply_pipeline`. Unknown arguments cause `TypeError` at call time. The `TestSerializationInterop.test_apply_with_kwargs_no_silent_ignore` test verifies forwarding works.

### 5.2 - `from_pil()` Has `linearize` Escape Hatch

```python
def from_pil(image, device="cpu", linearize: bool = True) -> Tensor:
    float_t = from_uint8(arr)
    return srgb_to_linear(float_t) if linearize else float_t
```

Users loading linear-light images can pass `linearize=False`.

---

## 6. Documentation Accuracy

### 6.1 - README Is Mostly Accurate

The README correctly states: *"All pipeline stages are implemented in pure PyTorch and are fully differentiable on both CPU and CUDA."* No false CUDA delegation claims.

The design decisions table accurately tracks fixes from earlier drafts, including the `from_pil(linearize=)` change, the serialization fix, and the dead CUDA import removal.

### 6.2 - Stale Module Docstring in `functional.py`

The module-level docstring still says:

> "Delegate math to the PPISP CUDA backend when available, with a pure-PyTorch fallback"

But the inline comment correctly says:

> "This module uses pure-PyTorch implementations for all pipeline stages."

No CUDA import exists in the file. The docstring is stale.

### 6.3 - `params.py` Comment Is Incorrect

```python
# This ensures PipelineParams() / ISPPipeline() is a true no-op.
```

`ISPPipeline()` is NOT a no-op (Section 3.1). Only `ISPPipeline.from_params(PipelineParams())` is.

---

## 7. Mathematical Correctness

### 7.1 - CRF Identity Math Is Correct

```python
_CRF_TAU_IDENTITY = math.log(math.expm1(0.7))
```

Verification: `softplus(log(expm1(0.7))) = log(1 + expm1(0.7)) = log(exp(0.7)) = 0.7`, so `tau = 0.3 + 0.7 = 1.0`. With tau=eta=1, xi=0.5, gamma=1, the S-curve reduces to `f(x) = x`.

### 7.2 - Preset Constraint Inversion Is Consistent

The `_make_crf_raw` function in `presets.py` and the `_CRF_TAU_IDENTITY` constants in `params.py` use different code paths but compute the same mathematical inverse. The `test_identity_preset_is_identity` test implicitly verifies this.

### 7.3 - `identity_crf_raw` Fixture Is Misleadingly Named

```python
@pytest.fixture
def identity_crf_raw():
    """CRF raw params that produce approximately tau=eta=1, xi=0.5, gamma=1."""
    zeros = torch.zeros(3)
    return zeros, zeros, zeros, zeros
```

Raw zeros produce gamma ≈ 0.793, which is not "approximately 1." The name suggests identity but delivers non-identity. Tests using this fixture only check range/shape/monotonicity (which work with any valid params), so no test logic is wrong - but the fixture name misleads developers.

---

## 8. Test Suite Assessment

### 8.1 - Comprehensive Coverage

| Test File | Tests | Notes |
|---|---|---|
| `test_functional.py` | 28 | Identity, known-value, shape, batch, gradient, warnings |
| `test_gradient_flow.py` | 6 | All four stages + gradient value verification |
| `test_modules.py` | 26 | **1 test fails** (Section 2) |
| `test_params.py` | 7 | Serialization, JSON validity |
| `test_pipeline_integration.py` | 28 | Identity invariant, serialization interop, end-to-end |
| `test_validation.py` | 16 | All validators, exception hierarchy |
| **Total** | **111** | |

### 8.2 - Strong Test Design

The `TestIdentityInvariant` class (4 tests) verifies the identity property through four different paths: `from_params`, the "identity" preset, `apply_pipeline`, and `easyppisp.apply()`. The `TestSerializationInterop` class (3 tests) catches the exact serialization format bug that plagued V1-V4.

The gradient value test remains excellent:
```python
expected_grad = img.sum().item() * torch.log(torch.tensor(2.0)).item()
assert abs(dt.grad.item() - expected_grad) < 1e-5
```

### 8.3 - Test Gaps

1. **`test_get_params_dict_constrained_values` is broken** - checks raw keys expecting physical values (Section 2)
2. **No monotonicity test with extreme params** - only tests at default zeros
3. **No CUDA tests** - `device` fixture only has `"cpu"`
4. **`test_load_default_is_identity`** - docstring says "identity" but only checks shape, not `torch.allclose`
5. **No test for `ISPPipeline()` vs `ISPPipeline.from_params(PipelineParams())`** - these differ and the difference is undocumented
6. **Unused fixture** - `img_hwc_f64` defined but never referenced

---

## 9. Missing Deliverables

### Implemented (strong coverage):

| Component | Quality |
|---|---|
| `params.py` with identity defaults | ✅ Pre-inverted CRF values |
| `validation.py` with 5 validators | ✅ Clear error messages |
| `functional.py` with 5 functions | ✅ Documented, differentiable |
| `color_homography.py` Eq. (6)-(12) | ✅ Constant tensors documented |
| `modules.py` with 5 classes | ✅ Independent, composable |
| `tasks.py` with 3 workflows | ✅ Thread-safe augmentation, proper serialization |
| `utils.py` with 10 functions | ✅ `linearize` escape hatch |
| `presets.py` with constraint inversion | ✅ `_make_crf_raw` |
| `__init__.py` with explicit `apply()` | ✅ No silent kwargs |
| `cli.py` with 2 subcommands | ✅ |
| Test suite: 111 tests in 7 files | ✅ (1 broken test) |

### Not implemented:

| Component | Impact |
|---|---|
| `ISPController` (Eq. 17, Sec 4.5) | Core paper feature |
| Regularization losses (Eq. 18-22) | Training support |
| `pyproject.toml` | Cannot install |
| CLI `augment` / `match` subcommands | Reduced CLI utility |
| CRF augmentation in `PhysicalAugmentation` | Incomplete physical variation |
| `docs/` directory | No standalone documentation |
| `examples/` directory | No runnable examples |
| `PipelineParams.from_constrained()` | Only `_make_crf_raw` in presets (private) |

---

## 10. Summary

| Criterion | Rating | Notes |
|---|---|---|
| **Importable** | ✅ | Clean import chain |
| **Identity invariant** | ⚠️ Split | `ISPPipeline.from_params(PipelineParams())` is identity; bare `ISPPipeline()` is not |
| **Mathematical correctness** | ✅ | CRF constraints, color ordering, vignetting coordinates all correct |
| **Test suite** | ⚠️ 1 broken test | `test_get_params_dict_constrained_values` fails due to key name change |
| **Serialization** | ✅ Fixed | `CameraMatchPair.save_params()` produces `PipelineParams`-compatible JSON |
| **API design** | ✅ | Explicit params, no silent swallowing, `linearize` escape hatch |
| **Code quality** | ✅ High | Docstrings, type hints, logging, error messages throughout |
| **Documentation accuracy** | ⚠️ | One stale docstring in `functional.py`; incorrect comment in `params.py` |
| **Completeness vs. prompt** | ~70% | Missing ISPController, regularization, pyproject.toml |
| **Production readiness** | ⚠️ Close | Fix the broken test, add pyproject.toml, reconcile `ISPPipeline()` vs `from_params` defaults |

The actionable issues are: (1) fix `test_get_params_dict_constrained_values` to read `_phys` keys, (2) reconcile `CameraResponseFunction()` default raws with `PipelineParams()` identity raws (or document the distinction), (3) correct the two stale documentation strings, and (4) add `pyproject.toml`.
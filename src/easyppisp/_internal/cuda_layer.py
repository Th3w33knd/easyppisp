"""
CUDA backend interface for EasyPPISP.

Handles JIT compilation or loading of the 'ppisp_cuda' extension.
Includes a pure-PyTorch fallback if CUDA is unavailable or compilation fails.

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import torch
import logging
import platform
import urllib.request
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Global flag to check if CUDA backend is available
_CUDA_AVAILABLE = False
_PPISP_CUDA = None

# Global flag to check if CUDA backend is available
_CUDA_AVAILABLE = False
_PPISP_CUDA = None
_VERSION = "0.1.0" # Should match pyproject.toml
_REPO_URL = "https://github.com/Th3w33knd/easyppisp"

def _get_binary_info() -> Tuple[str, str]:
    """Determine the platform-specific binary name and expected filename."""
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = sys.platform
    arch = platform.machine().lower()
    
    # Map architectures
    if arch in ['x86_64', 'amd64']:
        arch_str = 'x86_64'
    elif arch in ['arm64', 'aarch64']:
        arch_str = 'aarch64'
    else:
        arch_str = arch

    # CUDA tag
    cuda_ver = torch.version.cuda
    if cuda_ver:
        cuda_tag = f"cu{cuda_ver.replace('.', '')[:3]}" # e.g., cu118
    else:
        cuda_tag = "cpu"

    ext = ".so" if system != "win32" else ".dll"
    
    # The actual module name Python expects
    module_name = "ppisp_cuda"
    
    # The filename for storage/download
    filename = f"{module_name}-{_VERSION}-{py_ver}-{system}_{arch_str}-{cuda_tag}{ext}"
    
    return module_name, filename

def _download_remote_binary(dest_path: str) -> bool:
    """Attempt to download the pre-compiled binary from GitHub Releases."""
    _, filename = _get_binary_info()
    url = f"{_REPO_URL}/releases/download/v{_VERSION}/{filename}"
    
    logger.info(f"Attempting to download pre-compiled binary from {url}")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded {filename} to {dest_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download remote binary: {e}")
        return False

def _try_load_cuda():
    """Attempt to load or JIT-compile the CUDA extension."""
    global _CUDA_AVAILABLE, _PPISP_CUDA
    
    if not torch.cuda.is_available():
        return False

    try:
        # 1. Try importing if already installed/compiled
        # Add current directory to sys.path to help find local compiled extensions
        import sys
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if curr_dir not in sys.path:
            sys.path.insert(0, curr_dir) # Insert at the beginning to prioritize local

        import ppisp_cuda
        _PPISP_CUDA = ppisp_cuda
        _CUDA_AVAILABLE = True
        logger.info(f"Loaded pre-compiled EasyPPISP CUDA extension from {curr_dir}")
        return True
    except ImportError:
        # If direct import fails, remove the added path to avoid polluting sys.path
        if 'curr_dir' in locals() and curr_dir in sys.path:
            sys.path.remove(curr_dir)
        pass # Continue to JIT compilation attempt

    # 2. Try JIT compilation if source is available
    try:
        from torch.utils.cpp_extension import load
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_dir = os.path.join(curr_dir, "cuda")
        
        if os.path.exists(cuda_dir):
            sources = [
                os.path.join(cuda_dir, "src", "ppisp_impl.cu"),
                os.path.join(cuda_dir, "ext.cpp")
            ]
            # Check if sources exist
            if all(os.path.exists(s) for s in sources):
                logger.info("JIT compiling EasyPPISP CUDA extension...")
                _PPISP_CUDA = load(
                    name="ppisp_cuda",
                    sources=sources,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    verbose=False
                )
                _CUDA_AVAILABLE = True
                return True
    except Exception as e:
        logger.warning(f"Failed to JIT compile CUDA extension: {e}")

    # 3. Try Downloading from GitHub Releases
    try:
        module_name, filename = _get_binary_info()
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # For local execution, we want it to be importable as 'ppisp_cuda'
        # We'll save it as ppisp_cuda.so/dll in the current directory
        ext = ".so" if sys.platform != "win32" else ".dll"
        dest_filename = f"{module_name}{ext}" if sys.platform == "win32" else f"{module_name}.cpython-{sys.version_info.major}{sys.version_info.minor}-{platform.machine().lower()}-linux-gnu.so"
        
        # Actually, Python versioning in filenames is tricky.
        # Let's just use the standard name if we download it to the module dir.
        # On Linux, it's usually ppisp_cuda.cpython-310-x86_64-linux-gnu.so
        # If we download it as 'ppisp_cuda.so', Python might not find it if it expects the long name.
        # But we added curr_dir to sys.path, so 'import ppisp_cuda' should work if 'ppisp_cuda.so' exists.
        
        dest_path = os.path.join(curr_dir, f"{module_name}{ext}")
        
        if _download_remote_binary(dest_path):
            if curr_dir not in sys.path:
                sys.path.insert(0, curr_dir)
            import ppisp_cuda
            _PPISP_CUDA = ppisp_cuda
            _CUDA_AVAILABLE = True
            logger.info("Successfully loaded downloaded CUDA extension.")
            return True
    except Exception as e:
        logger.warning(f"Failed to load downloaded binary: {e}")

    return False

# Initialize on module load
_try_load_cuda()

def is_cuda_available() -> bool:
    """Returns True if the high-performance CUDA backend is ready."""
    return _CUDA_AVAILABLE

class PPISPCUDAFunction(torch.autograd.Function):
    """
    Autograd wrapper for the CUDA implementation.
    """
    @staticmethod
    def forward(ctx, exposure_params, vignetting_params, color_params, crf_params, 
                rgb_in, pixel_coords, resolution_w, resolution_h, camera_idx, frame_idx):
        
        # Ensure inputs are float32 and on CUDA
        exposure_params = exposure_params.contiguous().float()
        vignetting_params = vignetting_params.contiguous().float()
        color_params = color_params.contiguous().float()
        crf_params = crf_params.contiguous().float()
        rgb_in = rgb_in.contiguous().float()
        pixel_coords = pixel_coords.contiguous().float()

        rgb_out = _PPISP_CUDA.ppisp_forward(
            exposure_params, vignetting_params, color_params, crf_params,
            rgb_in, pixel_coords, resolution_w, resolution_h, camera_idx, frame_idx
        )

        ctx.save_for_backward(exposure_params, vignetting_params, color_params, crf_params, 
                              rgb_in, rgb_out, pixel_coords)
        ctx.resolution = (resolution_w, resolution_h)
        ctx.indices = (camera_idx, frame_idx)

        return rgb_out

    @staticmethod
    def backward(ctx, grad_rgb_out):
        (exposure_params, vignetting_params, color_params, crf_params, 
         rgb_in, rgb_out, pixel_coords) = ctx.saved_tensors
        res_w, res_h = ctx.resolution
        cam_idx, frm_idx = ctx.indices

        grad_rgb_out = grad_rgb_out.contiguous()

        grads = _PPISP_CUDA.ppisp_backward(
            exposure_params, vignetting_params, color_params, crf_params,
            rgb_in, rgb_out, pixel_coords, grad_rgb_out,
            res_w, res_h, cam_idx, frm_idx
        )
        
        # Unpack grads from tuple (v_exp, v_vig, v_col, v_crf, v_rgb_in)
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None, None, None, None

def ppisp_cuda(exposure_params, vignetting_params, color_params, crf_params, 
               rgb_in, pixel_coords, resolution_w, resolution_h, camera_idx, frame_idx):
    """EntryPoint for CUDA implementation."""
    return PPISPCUDAFunction.apply(
        exposure_params, vignetting_params, color_params, crf_params,
        rgb_in, pixel_coords, resolution_w, resolution_h, camera_idx, frame_idx
    )

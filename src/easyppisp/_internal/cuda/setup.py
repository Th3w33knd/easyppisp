# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import sys
import os

def get_cuda_arch_flags():
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list_env:
        arch_entries = arch_list_env.replace(";", " ").split()
        gencode_flags = []
        for entry in arch_entries:
            entry = entry.strip()
            if not entry: continue
            has_ptx = "+PTX" in entry
            arch_version = entry.replace("+PTX", "").strip()
            if "." in arch_version:
                major, minor = arch_version.split(".")
                compute_arch = f"compute_{major}{minor}"
                sm_arch = f"sm_{major}{minor}"
            else:
                compute_arch = f"compute_{arch_version}"
                sm_arch = f"sm_{arch_version}"
            if has_ptx: gencode_flags.append(f"-gencode=arch={compute_arch},code={compute_arch}")
            gencode_flags.append(f"-gencode=arch={compute_arch},code={sm_arch}")
        return gencode_flags, f"TORCH_CUDA_ARCH_LIST={arch_list_env}"

    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cap = torch.cuda.get_device_capability(device)
            arch = f"sm_{cap[0]}{cap[1]}"
            return [f"-arch={arch}"], arch
    except Exception: pass

    fallback_archs = [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    return fallback_archs, "fallback"

arch_flags, detected_arch = get_cuda_arch_flags()
nvcc_args = ["-O3", "--use_fast_math", "-lineinfo"] + arch_flags

setup(
    name="ppisp_cuda",
    ext_modules=[
        CUDAExtension(
            name="ppisp_cuda",
            sources=[
                "src/ppisp_impl.cu",
                "ext.cpp"
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_args}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Paths to the sources
curr_dir = os.path.dirname(os.path.abspath(__file__))
cuda_dir = os.path.join(curr_dir, "src", "easyppisp", "_internal", "cuda")
src_dir = os.path.join(cuda_dir, "src")

# Pixi environment CUDA paths
pixi_cuda = os.path.join(curr_dir, ".pixi", "envs", "gpu", "Library")
extra_include_paths = []

# Add targets/x64 include path (where cub/cub.cuh lives)
targets_include = os.path.join(pixi_cuda, "include", "targets", "x64")
if os.path.exists(targets_include):
    extra_include_paths.append(targets_include)
    print(f"Adding targets include path: {targets_include}")

setup(
    name='ppisp_cuda',
    ext_modules=[
        CUDAExtension(
            name='ppisp_cuda',
            sources=[
                os.path.join(src_dir, 'ppisp_impl.cu'),
                os.path.join(cuda_dir, 'ext.cpp'),
            ],
            extra_compile_args={
                'cxx': ['/O2'],
                'nvcc': ['-O3', '--use_fast_math']
            },
            include_dirs=extra_include_paths
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

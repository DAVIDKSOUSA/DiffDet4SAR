from glob import glob
from os import path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "detectron2", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob(path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob(path.join(extensions_dir, "**", "*.cu"), recursive=True)

    sources = [main_source] + [s for s in sources if s != main_source]
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or path.exists("/usr/local/cuda"):
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="diffdet4sar",
    version="0.1.0",
    description="DiffDet4SAR training and inference package",
    packages=find_packages(exclude=("datasets", "output*", "AIRCRAFT PRED")),
    include_package_data=True,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

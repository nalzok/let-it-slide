from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="rptc_kernels_cuda",
    ext_modules=[
        CUDAExtension(
            name="rptc_kernels",
            sources=["csrc/rptc_kernels/wrapper.cpp",
                     "csrc/rptc_kernels/inference.cu"],
            extra_compile_args={
                # "cxx": ["-O2", "-g", "-lineinfo", "-std=c++17"],
                # "nvcc": ["-O2", "-g", "-G", "-std=c++17", "--ptxas-options=-v", "-Xcompiler", "-rdynamic"],
                "cxx": ["-O3", "-lineinfo", "-std=c++17"],
                "nvcc": ["-O3", "-lineinfo", "-std=c++17", "--ptxas-options=-v"],
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)

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
                # "cxx": ["-g", "-lineinfo"],
                # "nvcc": ["-O2", "-g", "-Xcompiler", "-rdynamic", "--ptxas-options=-v", "-lineinfo", "-gencode=arch=compute_61,code=sm_61"],
                "cxx": ["-O3", "-std=c++17", "-lineinfo"],
                "nvcc": ["-O3", "-std=c++17", "--ptxas-options=-v", "-lineinfo", "-gencode=arch=compute_61,code=sm_61"],
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)

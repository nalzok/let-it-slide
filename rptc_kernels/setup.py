from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="rptc_kernels_cuda",
    ext_modules=[
        CUDAExtension(
            name="rptc_kernels",
            sources=["csrc/rptc_kernels/wrapper.cpp",
                     "csrc/rptc_kernels/inference.cu",
                     "csrc/rptc_kernels/inference_t.cu"],
            extra_compile_args={
                # "cxx": ["-O2", "--fast-math", "-g", "-lineinfo", "-std=c++17"],
                # "nvcc": ["-O2", "--use_fast_math", "-g", "-G", "-keep", "-std=c++17", "--ptxas-options=-v", "-Xcompiler", "-rdynamic"],
                "cxx": ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-keep", "-std=c++17", "--ptxas-options=-v"],
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)

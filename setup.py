# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import torch.cuda

from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
from pathlib import Path

ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'eden_utils', [
            'src/srrcomp/eden_utils/eden_utils.cpp',
            'src/srrcomp/eden_utils/csrc/cuda_hadamard.cu',
            'src/srrcomp/eden_utils/csrc/cuda_packing.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
    )

    ext_modules.append(extension)



this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='srrcomp',
    version='0.1.3',
    description='Structured random rotation (srr) based compression tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shay Vargaftik and Yaniv Ben-Itzhak',
    author_email='shayv@vmware.com, ybenitzhak@vmware.com',
    url='https://github.com/shayvar/structured-random-rotation-based-compression',
    license='BSD-3-Clause',
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    data_files = ['src/srrcomp/eden_utils/csrc/cuda_hadamard.h', 'src/srrcomp/eden_utils/csrc/cuda_packing.h', 'src/srrcomp/eden_utils/csrc/defs.h'],
    install_requires=['torch', 'numpy'],
    ext_package='srrcomp',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})

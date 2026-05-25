"""
Build script for _hot_paths Cython extension.

Usage:
    python setup_hot_paths.py build_ext --inplace

Requirements:
    pip install cython numpy

After building, hot_paths.py will automatically prefer the compiled .so/.pyd
over the pure Python fallback.
"""
from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

ext = Extension(
    name="_hot_paths",
    sources=["_hot_paths.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="hot_paths",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
        },
        annotate=True,          # generates _hot_paths.html showing Python overhead
    ),
)

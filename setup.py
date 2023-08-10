# Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>
#
# License: BSD 3 clause

from setuptools import setup, Extension, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys 

__version__ = "0.2.9"

with open('README.md', 'r') as f:
	long_description = f.read()

ext_modules = None 

if sys.platform == 'win32':
    print("Compiling for Windows")
    ext_modules = [
    	Pybind11Extension("_hierarchical_umap",
    		["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
    		language='c++',
    		extra_compile_args = [ '/openmp',  '/DINFO', '-IC:/Eigen'],
            extra_link_args = [ '/openmp', '/DINFO', '-IC:/Eigen'],
    		define_macros = [('VERSION_INFO', __version__)],
    		),

    ]
elif sys.platform == 'darwin':
    print("Compiling for MacOS")
    ext_modules = [
    Pybind11Extension("_hierarchical_umap",
        ["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
        language='c++',
        extra_compile_args = ['-O3', '-std=c++11', '-fPIC', '-fopenmp',  '-DINFO', '-I/usr/local/include'],
        extra_link_args = ['-O3', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO', '-I/usr/local/include'],
        define_macros = [('VERSION_INFO', __version__)],
        ),

    ]
else:
    print("Compiling for Linux")
    ext_modules = [
    Pybind11Extension("_hierarchical_umap",
        ["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
        language='c++',
        extra_compile_args = ['-O3', '-shared', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO'],
        extra_link_args = ['-O3', '-shared', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO'],
        define_macros = [('VERSION_INFO', __version__)],
        ),

    ]


setup(
    name="humap",
    version=__version__,
    author="Wilson E. Marcílio-Jr",
    author_email="wilson_jr@outlook.com",
    url="https://github.com/wilsonjr/humap",
    description="Hierarchical Uniform Manifold Approximation and Projection",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    license='MIT',
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    # install_requires=['numpy>=1.23.0', 'pybind11==2.10.1', 'scikit-learn>=1.1.3', 'scipy>=1.9.3'],
    packages=['humap'],
    zip_safe=False,
)

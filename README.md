=====
HUMAP
=====

Hierarchical Manifold Approximation and Projection (HUMAP) is a hierarchical dimensionality reduction technique 
based on `UMAP <http://github.com/lmcinnes/umap>`_ for non-linear dimensionality reduction. HUMAP allows you to:

1. Focus on important information while reducing the visual burden when exploring whole datasets;
2. Drill-down the hierarchy according to information demand.

The details of the algorithm can be found in our paper on `ArXiv <http://wilsonjr.github.io>_` or `TVCG <http://wilsonjr.github.io>`


-----------
Instalation
-----------

HUMAP was written in C++ for performance purposes and has a intuitive Python interface. 
It depends upon common machine learning libraries, such as ``scikit-learn`` and ``numpy``.
It also needs the ``pybind11`` due to the interface between C++ and Python.


Requirements:

* Python 3.6 or greater
* numpy
* scipy
* scikit-learn
* pybind11
* Eigen (C++)


For Windows:

The `Eigen <https://eigen.tuxfamily.org/>`_ library does not have to be installed. Just add the files to `C:\Eigen` or use the manual installation.

Manual instalation: 

For manually installing HUMAP, download the project and proceed as follows:

1. ``python setup.py bdist_wheel``
2. ``pip install dist/humap*.whl``


--------------
Usage examples
--------------



--------
Citation
--------


-------
License
-------

HUMAP follows the 3-clause BSD license.


HUMAP uses the open-source NNDescent implementation from `EFANNA <https://github.com/ZJULearning/efanna>`. 

HUMAP uses `UMAP <http://github.com/lmcinnes/umap>`_ for embedding hierarchy levels, this project would not be possible 
without UMAP amazing technique and package.


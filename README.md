.. -*- mode: rst -*-

|conda_version|_ |conda_downloads|_ |pypi_version|_ |pypi_downloads|_

.. |pypi_version| image:: https://img.shields.io/pypi/v/humap.svg
.. _pypi_version: https://pypi.python.org/pypi/humap/

.. |pypi_downloads| image:: https://pepy.tech/badge/humap
.. _pypi_downloads: https://pepy.tech/project/humap

.. |conda_version| image:: https://anaconda.org/conda-forge/humap/badges/version.svg
.. _conda_version: https://anaconda.org/conda-forge/humap

.. |conda_downloads| image:: https://anaconda.org/conda-forge/humap/badges/downloads.svg
.. _conda_downloads: https://anaconda.org/conda-forge/humap

.. image:: images/humap-2M.gif
	:alt: HUMAP exploration on Fashion MNIST dataset

=====
HUMAP
=====

Hierarchical Manifold Approximation and Projection (HUMAP) is a technique based on `UMAP <https://github.com/lmcinnes/umap/>`_ for hierarchical dimensionality reduction. HUMAP allows to:


1. Focus on important information while reducing the visual burden when exploring huge datasets;
2. Drill-down the hierarchy according to information demand.

The details of the algorithm can be found in our paper on `ArXiv <https://arxiv.org/abs/2106.07718>`_. This repository also features a C++ UMAP implementation.


-----------
Installation
-----------

HUMAP was written in C++ for performance purposes, and provides an intuitive Python interface. It depends upon common machine learning libraries, such as ``scikit-learn`` and ``NumPy``. It also needs the ``pybind11`` due to the interface between C++ and Python.


Requirements:

* Python 3.6 or greater
* numpy
* scipy
* scikit-learn
* pybind11
* pynndescent (for reproducible results)
* Eigen (C++)

If you have these requirements installed, use PyPI:

.. code:: bash

    pip install humap
    
Alternatively (and preferable), you can use conda to install:

.. code:: bash

    conda install humap


**If using pip**:

HUMAP depends on `Eigen <https://eigen.tuxfamily.org/>`_. Thus, make it sure to place the headers in **/usr/local/include** if using Unix or **C:\\Eigen** if using Windows.

**Manual installation**: 

For manually installing HUMAP, download the project and proceed as follows:

.. code:: bash
 	
 	python setup.py bdist_wheel

.. code:: bash

 	pip install dist/humap*.whl


--------------
Usage examples
--------------

The simplest usage of HUMAP is as it follows:

**Fitting the hierarchy**

.. code:: python

	import humap
	from sklearn.datasets import fetch_openml


	X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	# build a hierarchy with three levels
	hUmap = humap.HUMAP([0.2, 0.2])
	hUmap.fit(X, y)

	# embed level 2
	embedding2 = hUmap.transform(2)

Refer to *notebooks/* for complete examples.

**C++ UMAP implementation**

You can also fit a one-level HUMAP hierarchy, which essentially fits UMAP projection.

.. code:: python

	umap_reducer = humap.UMAP()
	embedding = umap_reducer.fit_transform(X)

--------
Citation
--------

Please, use the following reference to cite HUMAP in your work:

.. code:: bibtex

	@ARTICLE{marciliojr_humap2024,
		author={Marcílio-Jr, Wilson E. and Eler, Danilo M. and Paulovich, Fernando V. and Martins, Rafael M.},
		journal={IEEE Transactions on Visualization and Computer Graphics}, 
		title={HUMAP: Hierarchical Uniform Manifold Approximation and Projection}, 
		year={2024},
		volume={},
		number={},
		pages={1-10},
		doi={10.1109/TVCG.2024.3471181}
	}


-------
License
-------

HUMAP follows the 3-clause BSD license.


......

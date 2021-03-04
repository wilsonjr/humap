.. -*- mode: rst -*-

=====
HUMAP
=====

Hierarchical Manifold Approximation and Projection (HUMAP) is a hierarchical dimensionality reduction technique 
based on `UMAP <https://github.com/lmcinnes/umap/>`_ for non-linear dimensionality reduction. HUMAP allows you to:

1. Focus on important information while reducing the visual burden when exploring whole datasets;
2. Drill-down the hierarchy according to information demand.

The details of the algorithm can be found in our paper on `ArXiv <https://wilsonjr.github.io>`_ or `TVCG <https://wilsonjr.github.io>`_.


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

If you have these requirements installed, use PyPI:

.. code:: bash

    pip install umap-learn


**For Windows**:

The `Eigen <https://eigen.tuxfamily.org/>`_ library does not have to be installed. Just add the files to `C:\\Eigen` or use the manual installation.

**Manual instalation**: 

For manually installing HUMAP, download the project and proceed as follows:

.. code:: bash
 	
 	python setup.py bdist_wheel

.. code:: bash

 	pip install dist/humap*.whl


--------------
Usage examples
--------------

HUMAP package follows the same idea of sklearn classes, in which you need to fit and transform data.

.. code:: python

	import humap
	from sklearn.datasets import fetch_openml


	X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	hUmap = humap.HUMAP()
	hUmap.fit(X, y)


By now, you can control six parameters that are related to the hierarchy construction and the embedding performed by UMAP.


There are a number of parameters that can be set for the UMAP class; the
major ones are as follows:

 -  ``levels``: Controls the number of hierarchical levels + the first one (whole dataset).
 	This parameters also controls how many data points are in each hierarchical level.
 	The default is ``[0.2, 0.2]``, meaning the HUMAP will produce three levels: The first one
 	with the whole dataset, the second one with 20% of the first level, and the third with 20%
 	of the second level.

 -  ``n_neighbors``: This parameter controls the number of neighbors for approximating
 	the manifold structures. Larger values produce embedding that preserves more of the 
 	global relations. In HUMAP, we recommend and set the default value to be ``100``.

 -  ``min_dist``: This parameter, used in UMAP dimensionality reduction, controls the allowance
 	to cluster data points together. According to UMAP documentation, larger values allows evenly
 	distributed embeddings, while smaller values encodes better the local structures. 
 	We set this parameter as 0.15 as default.

 -  ``knn_algorithm``: Controls which knn approximation will be used, in which NNDescent is the default.
 	Another options are ANNOY or FLANN if you have Python installations of these algorithms, at the expense of
 	slower run-time executions compared with NNDescent.

 -  ``init``: Controls the method for initing the low-dimensional representation. We set ``Spectral`` as default 
 	since it yields better global structures preservation. You can also use ``random`` initialization.

 -  ``verbose``: Controls the verbosity of the algorithm.



--------
Citation
--------

Please, use the following reference to cite HUMAP in your work:

.. code:: bibtex

    @article{MarcilioJr2021_HUMAP,
      title={Hierarchical Uniform Manifold Approximation and Projection},
      author={Marcílio-Jr, W. E. and Eler, D. M. and Paulovich, F. V.},
      journal={IEEE Transations on Visualization and Computer Graphics},
      volume={},
      number={},
      pages={},
      year={2021}
    }


-------
License
-------

HUMAP follows the 3-clause BSD license.


HUMAP uses the open-source NNDescent implementation from `EFANNA <https://github.com/ZJULearning/efanna>`_. It uses `UMAP <http://github.com/lmcinnes/umap>`_ for embedding hierarchy levels, this project would not be possible 
without UMAP amazing technique and package.


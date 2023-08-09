# Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>
#
# License: BSD 3 clause

import _hierarchical_umap
import numpy as np 

from scipy.optimize import curve_fit
from sklearn.utils import check_array

import logging

class HUMAP(object):
	"""
	Class for wrapping the pybind11 interface of HUMAP C++ implementation
	
	...

	Parameters
	----------
	levels (np.array): shape (n_levels-1) (optinal, default [0.2, 0.2])
		A numpy array to inform the percentage of data points in each hierarchy level starting from the second .

	n_neighbors (int): (optional, default 100)
		The number of neighbors using for k nearest neighbor computation.

	min_dist (float): (optional, default 0.15)
		The effective minimum distance between embedded points for UMAP technique.

	knn_algorithm (str): (optional, default 'NNDescent')
		The kNN algorithm used for affinity computation. Options include:
			* NNDescent
			* KDTree_NNDescent
			* ANNOY (Python instalation required)
			* FLANN (Python instalation required)


	init (str): (optional, default 'Random')
		Initialization method for the low dimensional embedding. Options include:	
			* Spectral 
			* random

	reproducible (bool): (optional, default 'False')
		If the results among different runs need to be reproducible. It affects the runtime execution.

	verbose (bool): (optional, default False)
		Controls logging.

	"""
	def __init__(self, levels=np.array([0.2, 0.2]), n_neighbors=100, min_dist=0.15, knn_algorithm='NNDescent', init="Random", verbose=False, reproducible=False):

		self.levels = levels
		self.n_levels = len(levels)+1
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.knn_algorithm = knn_algorithm
		self.verbose = verbose
		self.init = init
		self.reproducible = reproducible
		self.h_umap = _hierarchical_umap.HUMAP('precomputed', self.levels, self.n_neighbors, self.min_dist, self.knn_algorithm, self.init, self.verbose, self.reproducible)


	def fit(self, X, y=None):
		"""
		Fits a HUMAP hierarchy
		
		Parameters
		----------
		X (np.array): shape (n_samples, n_features)
			The dataset consisting of n data points by m features

		y (np.array): shape (n_samples) (optinal, default None)
			The dataset labels

		Raises
		------
		ValueError
			If X:
				* is None 
				* is not a Numpy array
				* is not a two-dimensional array
		"""

		if X is None:
			raise ValueError("X must be a valid array")

		if not isinstance(X, np.ndarray):
			raise ValueError("X must be a numpy array")

		if len(X.shape) != 2:
			raise ValueError("X must be a two-dimensional array")

		if X.shape[1] <= 2:
			raise ValueError("X.shape[1] must be n-dimensional array (n > 2)")

		if y is None:
			y = np.zeros(X.shape[0])

		N = X.shape[0]
		for i, pct_level in enumerate([1.0] + self.levels.tolist()):
			if self.n_neighbors > int(pct_level * N):
				raise ValueError("Cannot induce a hierarchy since {} > {} on level {}, consider decreasing n_neighbors.".format(self.n_neighbors, int(pct_level * N), i))
			N *= pct_level


		X = check_array(X, dtype=np.float32, accept_sparse='csr', order='C')
		a, b = self.find_ab_params(1.0, self.min_dist)
		self.h_umap.set_ab_parameters(a, b)

		self.h_umap.fit(X, y)

	def set_focus_context(self, focus_context):
		r"""
		Defines how th embedding will be performed in terms of visualization

			Focus+Context (true) means that lower hierarchical levels will be projected
		together with higher hierarchical levels

	
		Parameters
		----------
		focus_context (bool): indicates if the subsets of data will be projected based focus+context approach
		"""
		self.h_umap.set_focus_context(focus_context)


	def set_influence_neighborhood(self, n_neighbors):
		r"""
		Defines how much of the neighborhood will be used in similarity computation.
		It adds local information to the resulting embedding.

		Parameters
		----------
		n_neighbors (int): the number of local neighbors used in similarity computation.
		"""

		self.h_umap.set_influence_neighborhood(n_neighbors)


	def original_indices(self, level):
		r"""
		Returns the original indices of the data points in a hierarchical level.
        
		Parameters
		----------

		level (int): the level of interest.

		Returns
		-------
		np.array: the indices of each data point in the level passed as parameter.
		"""

		return self.h_umap.get_original_indices(level)

	def transform_with_init(self, level, X_embedded):
		return self.h_umap.transform_with_init(level, X_embedded)

	def transform(self, level, **kwargs):
		r"""
		Generates the embedding for a given hierarchy level.
		This method is used to embed:
			* a hierarchical level, when passing just a level as parameter
			* a subset of classes, when passing an array with class labels and class_based = True
			* a subset of data points, when passing an array with indices and class_based = False
		
		Parameters
		----------
		level (int): the hierarchical level to embed.

		**kwargs (dict):
			* indices (np.array): indices of data points of interest or class labels.
			* class_based (bool): specifies if the embed is based on classes or indices.

		Raises
		------
		TypeError
			If the parameters of 'kwargs' diverge from 'indices' and 'class_based'.	

		Returns
		-------
		if kwargds == None
			np.array: The embedded hierarchy level
		else
			tuple:
				np.array: The embedded subset of the hierarchy level
				np.array: The labels of the embedded subset
				np.array: The indices of the subset on the hierarchy level

		"""

		if len(kwargs) == 0:
			return self.h_umap.transform(level)
		else:

			try:	
				embedding = None 

				if len(kwargs) == 1 or kwargs['class_based'] == False:
					embedding = self.h_umap.project_indices(level, kwargs['indices'])
				else:
					embedding = self.h_umap.project(level, kwargs['indices'])

				y = self.h_umap.get_labels_selected()
				indices_cluster = self.h_umap.get_indices_selected() 
				indices_fixed = self.h_umap.get_indices_fixed()
				return [embedding, y, indices_cluster, indices_fixed]

			except:
				raise TypeError("Accepted parameters: indices and class_based.")

	def labels(self, level):
		r"""
		Gets the labels of a particular hierarchy level

		Parameters
		----------
		level (int): the level of interest.

		Raises
		------
		ValueError
			If level equals 0 or greater than the highest level.			

		Returns
		-------
		np.array: the labels for the data points in the specified level
		"""

		if level <= 0 or level >= self.n_levels:
			raise ValueError("level must be in [1, n_levels-1]")
		else:			
			return self.h_umap.get_labels(level)


	def fix_datapoints(self, datapoints):
		r"""
		Data points used to guide Stochastic Gradient Descent on the mental map preservation of subsequent projections (of hierarchy levels)

		Parameters
		----------
		datapoints (np.array): The data points already projected hierarchy levels.

		Raises
		------
		ValueError 
			If datapoints is not a two-dimensional array
		
		"""

		if len(datapoints.shape) != 2:
			raise ValueError("Fix data points must be two-dimensional")

		self.h_umap.set_fixed_datapoints(datapoints)

	def set_fixing_term(self, fixing_term):
		r"""
		Fixing term used to preserve the mental map of subsequent projections (of hierarchy levels)

		Parameters
		----------
		fixing_term (float): The fixing term (between 0 and 1) on how the data points used to guide mental map preservation will be free during SGD optimization.

		Raises
		------
		ValueError 
			If fixing_term < 0 or fixing_term > 1
		
		"""
		if fixing_term < 0 or fixing_term > 1.0:
			raise ValueError("Fixing term must be between 0 and 1")

		self.h_umap.set_fixing_term(fixing_term)


	def set_info_file(self, info_file=""):
		self.h_umap.set_info_file(info_file)

	def set_n_epochs(self, epochs):
		self.h_umap.set_n_epochs(epochs)

	def get_knn(self, level):

		if level < 0 or level >= self.n_levels:
			raise ValueError("level must be in [0, n_levels-1]")
		else:			
			return self.h_umap.get_knn(level)

	def get_knn_distances(self, level):

		if level < 0 or level >= self.n_levels:
			raise ValueError("level must be in [0, n_levels-1]")
		else:			
			return self.h_umap.get_knn_dists(level)


	def influence(self, level):
		r"""
		Gets the information on how each landmark influence on the subsequent level

		Parameters
		----------
		level (int): The current level of landmarks

		Returns
		-------
		np.array: An array of integers containing how many data points each landmark influence on the subsequent level

		"""
		return self.h_umap.get_influence(level)


	def influence_selected(self):
		r"""
		Gets the information on how each selected landmark (when projecting subsets) influence on the subsequent level

		Returns
		-------
		np.array: An array of integers containing how many data points each landmark influence on the subsequent level

		"""
		return self.h_umap.get_influence_selected()
	
	def find_ab_params(self, spread, min_dist):
		"""
			From UMAP official implementation: https://github.com/lmcinnes/umap		
		"""
		def curve(x, a, b):
			return 1.0 / (1.0 + a * x ** (2 * b))

		xv = np.linspace(0, spread * 3, 300)
		yv = np.zeros(xv.shape)
		yv[xv < min_dist] = 1.0
		yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
		params, covar = curve_fit(curve, xv, yv)
		return params[0], params[1]


class UMAP(HUMAP):
	"""
	Class for wrapping the pybind11 interface of HUMAP C++ implementation
	
	...

	Parameters
	----------
	n_neighbors (int): (optional, default 100)
		The number of neighbors using for k nearest neighbor computation.

	min_dist (float): (optional, default 0.15)
		The effective minimum distance between embedded points for UMAP technique.

	knn_algorithm (str): (optional, default 'NNDescent')
		The kNN algorithm used for affinity computation. Options include:
			* NNDescent
			* KDTree_NNDescent
			* ANNOY (Python instalation required)
			* FLANN (Python instalation required)

	init (str): (optional, default 'Spectral')
		Initialization method for the low dimensional embedding: Options include:	
			* Spectral
			* random

	verbose (bool): (optional, default False)
		Controls logging.

	"""
	def __init__(self, n_neighbors=100, min_dist=0.15, knn_algorithm='NNDescent', init="Spectral", verbose=False, reproducible=False):
		super().__init__(np.array([]), n_neighbors, min_dist, knn_algorithm, init, verbose, reproducible)

	def fit_transform(self, X, X_embedded=None):
		"""
		Generates the embedding.
		
		Parameters
		----------
		X (np.array): The dataset consisting of n data points by m features

		Raises
		------
		ValueError
			If X:
				* is None 
				* is not a Numpy array
				* is not a two-dimensional array

		"""

		if X.shape[1] <= 2:
			raise ValueError("Input dimensionality must be > 2.")

		super().fit(X, None)
		if X_embedded is None:
			return super().transform(0)	
				
		return super().transform_with_init(0, X_embedded)	

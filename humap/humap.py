import _hierarchical_umap
import numpy as np 

from scipy.optimize import curve_fit

from sklearn.utils import check_array

class HUMAP(object):
	"""
	Class for wrapping the pybind11 interface of HUMAP C++ implementation
	
	...

	Parameters
	----------
	levels : array, shape (n_levels-1)
		A numpy array to inform the percentage of data points in each hierarchy level starting from the second .

	n_neighbors : int (optional, default 100)
		The number of neighbors using for k nearest neighbor computation.

	min_dist : float (optional, default 0.15)
		The effective minimum distance between embedded points for UMAP technique.

	knn_algorithm : str (optional, default 'NNDescent')
		The kNN algorithm used for affinity computation. Options include:
			* NNDescent
			* KDTree_NNDescent
			* ANNOY (Python instalation required)
			* FLANN (Python instalation required)

	init : str (optional, default 'Spectral')
		Initialization method for the low dimensional embedding: Options include:	
			* Spectral
			* random

	verbose : bool (optional, default True)
		Controls logging.

	"""
	def __init__(self, levels, n_neighbors=100, min_dist=0.15, knn_algorithm='NNDescent', init="Spectral", verbose=True):
		self.levels = levels
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.knn_algorithm = knn_algorithm
		self.verbose = verbose
		self.init = init

		self.h_umap = _hierarchical_umap.HUMAP('precomputed', self.levels, self.n_neighbors, self.min_dist, self.knn_algorithm, self.init, self.verbose)


	def fit(self, X, y=None):
		"""
		Fits a HUMAP hierarchy
		
		Parameters
		----------
		X : array, shape (n_samples, n_features)
			The dataset consisting of n data points by m features

		y : array, shape (n_samples) (optinal, default None)
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

		if y is None:
			y = np.zeros(X.shape[0])


		X = check_array(X, dtype=np.float32, accept_sparse='csr', order='C')
		a, b = self.find_ab_params(1.0, self.min_dist)
		self.h_umap.set_ab_parameters(a, b)

		self.h_umap.fit(X, y)



	def set_influence_neighborhood(self, n_neighbors):
		r"""
		Defines how much of the neighborhood will be used in similarity computation.
		It adds local information to the resulting embedding.

		Parameters
		----------
		n_neighbors : int 
			The number of local neighbors used in similarity computation.
		"""

		self.h_umap.set_influence_neighborhood(n_neighbors)


	def original_indices(self, level):
		r"""
		Returns the original indices of the data points in a hierarchical level.
        
		Parameters
		----------

		level : int
			The level of interest.

		Returns
		-------

		array
			An array with the original indices of each data point in the level passed as parameter.
		"""

		return self.h_umap.get_original_indices(level)

	def transform(self, level, **kwargs):
		r"""
		Generates the embedding for a given hierarchy level.
		This method is used to embed:
			* a hierarchical level, when passing just a level as parameter
			* a subset of classes, when passing an array with class labels and class_based = True
			* a subset of data points, when passing an array with indices and class_based = False
		
		Parameters
		----------
		level : int
			The hierarchical level to embed.

		\**kwargs : dict
			* indices : array
				Indices of data points of interest or class labels.
			* class_based : bool
				Specifies if the embed is based on classes or indices.


		Raises
		------
		TypeError
			If the parameters of 'kwargs' diverge from 'indices' and 'class_based'.	
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
				return [embedding, y, indices_cluster]

			except:
				raise TypeError("Accepted parameters: indices and class_based.")

	def labels(self, level):
		r"""
		Gets the labels of a particular hierarchy level

		Parameters
		----------
		level : int
			The level of interest.

		Raises
		------
		ValueError
			If level equals 0 or greater than the highest level.			
		"""


		try:
			return self.h_umap.get_labels(level)
		except:
			raise ValueError("level must be in [1, n_levels-1]")

	
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

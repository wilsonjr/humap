import _hierarchical_umap
import numpy as np 

from scipy.optimize import curve_fit

from sklearn.utils import check_array

class HUMAP(object):
	
	def __init__(self, levels, n_neighbors=100, min_dist=0.15, knn_algorithm='NNDescent', verbose=True):
		self.levels = levels
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.knn_algorithm = knn_algorithm
		self.verbose = verbose


	def fit(self, X, y):

		X = check_array(X, dtype=np.float32, accept_sparse='csr', order='C')
		
		self.h_umap = _hierarchical_umap.HUMAP('precomputed', self.levels, self.n_neighbors, self.min_dist, self.knn_algorithm, self.verbose)
		
		a, b = self.find_ab_params(1.0, self.min_dist)
		self.h_umap.set_ab_parameters(a, b)

		self.h_umap.fit(X, y)

	def original_indices(self, level):
		return self.h_umap.get_original_indices(level)

	def transform(self, level, **kwds):
		if len(kwds) == 0:
			return self.h_umap.transform(level)
		else:

			try:	
				embedding = None 

				if len(kwds) == 1 or kwds['class_based'] == False:
					embedding = self.h_umap.project_indices(level, kwds['indices'])
				else:
					embedding = self.h_umap.project(level, kwds['indices'])

				y = self.h_umap.get_labels_selected()
				indices_cluster = self.h_umap.get_indices_selected() 
				return [embedding, y, indices_cluster]

			except:
				raise TypeError("Accepted parameters: indices and class_based")

	def labels(self, level):
		return self.h_umap.get_labels(level)

	"""
		From UMAP official implementation: https://github.com/lmcinnes/umap		
	"""
	def find_ab_params(self, spread, min_dist):
		def curve(x, a, b):
			return 1.0 / (1.0 + a * x ** (2 * b))

		xv = np.linspace(0, spread * 3, 300)
		yv = np.zeros(xv.shape)
		yv[xv < min_dist] = 1.0
		yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
		params, covar = curve_fit(curve, xv, yv)
		return params[0], params[1]

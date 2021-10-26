import humap

import unittest

import numpy as np

from sklearn.datasets import fetch_openml 
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

class TestUmap(unittest.TestCase):

    def setUp(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)        
        X = normalize(X)

        self.X = X 
        self.y = y
    
    def tearDown(self):
        pass 

    
    def test_numberDataPoints(self):
        reducer = humap.UMAP(n_neighbors=15)
        embedding = reducer.fit_transform(self.X)
        
        self.assertEqual(embedding.shape, (len(self.X), 2), "incorrect embedding size")


    def test_dimensionality1(self):

        X = np.random.rand(1000, 1)
        reducer = humap.UMAP(n_neighbors=15)

        self.assertRaises(ValueError, reducer.fit_transform, X)

    def test_dimensionality2(self):

        X = np.random.rand(1000, 2)
        reducer = humap.UMAP(n_neighbors=15)

        self.assertRaises(ValueError, reducer.fit_transform, X)




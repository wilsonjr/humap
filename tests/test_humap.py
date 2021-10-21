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

    
    def test_influenceHierarchyLevels(self):
        reducer = humap.HUMAP(n_neighbors=15)
        reducer.fit(self.X)

        level2 = reducer.transform(2)
        level1 = reducer.transform(1)
        level0 = reducer.transform(0)

        influence2to1 = reducer.influence(2)
        influence1to0 = reducer.influence(1)

        self.assertEqual((level1.shape[1], level0.shape[1]), (np.sum(influence2to1), np.sum(influence1to0)), "missing influence on hierarchy levels")

    def test_lessPointsOnTop(self):

        reducer = humap.HUMAP(n_neighbors=15)
        reducer.fit(self.X)


        level2 = reducer.transform(2)
        level1 = reducer.transform(1)
        level0 = reducer.transform(0)

        self.assertLess(level1.shape[1], level2.shape[2])
        self.assertLess(level1.shape[0], level2.shape[1])

    def test_dimensionality2(self):

        X = np.random.rand(1000, 1)
        reducer = humap.HUMAP(n_neighbors=15)

        self.assertRaises(ValueError, lambda: reducer.fit(X), "input dimensionality <= 2")

    def test_noneArray(self):
        reducer = humap.HUMAP(n_neighbors=15)

        self.assertRaises(ValueError, lambda: reducer.fit(None), "input array must be a valid array")


    def test_shapeArray(self):

        X = np.random.rand(1000, 2, 3)
        reducer = humap.HUMAP(n_neighbors=15)

        self.assertRaises(ValueError, lambda: reducer.fit(X), "input array must be two-dimensional")

    def test_nNeighborsGreaterThanLevel(self):

        X = np.random.rand(1000, 2)
        reducer = humap.HUMAP(n_neighbors=100)

        self.assertRaises(ValueError, lambda: reducer.fit(X), "input array must be two-dimensional")

    def test_ndArray(self):

        X = np.random.rand(1000, 2).tolist()
        reducer = humap.HUMAP(n_neighbors=10)

        self.assertRaises(ValueError, lambda: reducer.fit(X), "input array must be two-dimensional")




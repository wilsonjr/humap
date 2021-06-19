import humap
import matplotlib.pyplot as plt 

from sklearn.datasets import fetch_openml 
from sklearn.preprocessing import normalize

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = normalize(X)

reducer = humap.UMAP(n_neighbors=15)
embedding = reducer.fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1], c=y.astype(int), cmap='viridis')
plt.show()


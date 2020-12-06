import numpy as np 
import pandas as pd
import seaborn as sns
import hierarchical_umap as h_umap 
from scipy.sparse import load_npz, csr_matrix
import time
import umap
import matplotlib.pyplot as plt
import demap
import math

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
# from sklearn.manifold import trustworthiness
from sklearn.utils import check_random_state, check_array





def scale(value, leftMin, leftMax, rightMin, rightMax):

	leftSpan = leftMax - leftMin 
	rightSpan = rightMax - rightMin

	valueScaled = float(value - leftMin) / float(leftSpan)

	return rightMin + (valueScaled * rightSpan)


def transform_sizes(values, minValue, maxValue, rightMin=3, rightMax=50):


	areas = []
	for value in values:
		areas.append(scale(value, minValue, maxValue, rightMin, rightMax))

	areas = np.array(areas)

	return (4.*areas)/math.pi 


def compute_trustworthiness(X, X_emb, Khigh=30):

	ks = np.zeros(Khigh)
	trust = np.zeros(Khigh)
	for i in tqdm(range(1, Khigh+1)):
		ks[i-1] = i
		trust[i-1] = trustworthiness(X, X_emb, n_neighbors=i)

	return ks, trust

def NNP(X, X_emb, Khigh=30):
	neigh_high = NearestNeighbors(n_neighbors=Khigh, n_jobs=-1)
	neigh_high.fit(X)
	high_dists, high_indices = neigh_high.kneighbors(X)

	
	neigh_emb = NearestNeighbors(n_neighbors=Khigh, n_jobs=-1)
	neigh_emb.fit(X_emb)
	emb_dists, emb_indices = neigh_emb.kneighbors(X_emb)	

	m_precision = np.zeros(Khigh)
	m_recall = np.zeros(Khigh)
	for i in tqdm(range(X.shape[0])):
		high_current = high_indices[i][1:]
		for k in range(1, Khigh+1):
			emb_current = emb_indices[i][1:k+1]            

			tp = len(np.intersect1d(high_current, emb_current))

			precision_val = float(tp)/k
			recall_val = float(tp)/Khigh

			m_precision[k-1] += precision_val
			m_recall[k-1] += recall_val

	m_precision = m_precision/float(X.shape[0])
	m_recall = m_recall/float(X.shape[0])


	return m_precision, m_recall

def neighborhood_preservation(X, X_emb, Khigh=30):
    
    neigh_high = NearestNeighbors(n_neighbors=Khigh+1, n_jobs=-1)
    neigh_high.fit(X)
    high_dists, high_indices = neigh_high.kneighbors(X)


    neigh_emb = NearestNeighbors(n_neighbors=Khigh+1, n_jobs=-1)
    neigh_emb.fit(X_emb)
    emb_dists, emb_indices = neigh_emb.kneighbors(X_emb)

    npres = np.zeros(Khigh)
    
    for k in range(1, Khigh+1):
        for i in range(X.shape[0]):
            high_current = high_indices[i][1:k+1]
            emb_current = emb_indices[i][1:k+1]
            
            tp = len(np.intersect1d(high_current, emb_current))
            
            npres[k-1] += (tp/k)
        
        
    npres /= float(X.shape[0])
    
    return npres

def get_size(value):

	area = (math.PI/4.0)*1



level = 0

n = 5000
n_neighbors = 15


fashionTrain = pd.read_csv('data/fashion-train.csv')

fashionX = fashionTrain.values[:,2:]
fashionY = fashionTrain.values[:, 1].astype(int)

print(fashionX.shape, fashionY.shape)

X = fashionX
y = fashionY

# X = np.load('./data/MNIST_70000.npy')
# y = np.load('./data/MNIST_70000_label.npy').astype(int)
# X = normalize(X)
# print(greatest.shape, cols.shape, graph.shape, knn_dists.shape)
X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
print(X.shape)

# cpp_umap = h_umap.UMAP("euclidean", "FAISS_IVFFlat")
# start = time.time()
# cpp_umap.fit_hierarchy(mnist)

# end = time.time()
# print("time: %.5fs" % (end-start))

hUmap = h_umap.HUMAP("precomputed", np.array([0.186083333, 0.177698164]), 15, "FLANN", True)
hUmap.fit(X, y)


second_level = X[hUmap.get_original_indices(1),:]#X[hUmap.get_indices(0),:]
third_level = X[hUmap.get_original_indices(2),:]#second_level[hUmap.get_indices(1),:]
second_indices = hUmap.get_indices(0)
third_indices = hUmap.get_indices(1)

print("second_indices")
print(second_indices[:30])
print("third indices")
print(third_indices[:30])

print("second_level: ", second_level.shape)
print("third_level: ", third_level.shape)

print("second_level: ", hUmap.get_indices(0).shape)
print("third_level: ", hUmap.get_indices(1).shape)




y2 = hUmap.get_labels(2)
embedding2 = hUmap.get_embedding(2)







# indices1 = hUmap.get_indices(1)
# plt.scatter(embedding1[indices1, 0], embedding1[indices1, 1], c ='red', alpha=1, s=1)



# precision_third, recall_third = compute_trustworthiness(third_level, embedding2)
# precision_second, recall_second = compute_trustworthiness(second_level, embedding1)


# plt.plot(precision2, recall2)
# plt.show()


# plt.plot(precision, recall)
# plt.show()


influence2 = hUmap.get_influence(2)
influence1 = hUmap.get_influence(1)
maxValue = max(np.max(influence1), np.max(influence2))
print(np.max(influence2), np.max(influence1))
print(influence2)
print("influence2")
print(np.sum(influence2), np.sum(influence2==0))
print(np.sum(influence1), np.sum(influence1==0))

# influence1 = hUmap.get_influence(1)
# print("\n\ninfluence1")
# maxValue = np.max(influence1)
# print(np.sum(influence1), np.sum(influence1==0))
	

# plt.plot(precision_third, recall_third)
# plt.ylim(0, 1)
# plt.show()

# plt.plot(precision_second, recall_second)
# plt.ylim(0, 1)
# plt.show()


s2 = transform_sizes(influence2, 1, maxValue, rightMin=8, rightMax=300)
s1 = transform_sizes(influence1, 1, maxValue, rightMin=8, rightMax=300)
s0 = transform_sizes([1]*len(y), 1, maxValue, rightMin=8, rightMax=300)

print("s2")
print(s2[:10])

print("\ns1")
print(s1[:10])

print("\ns0")
print(s0[:10])


# plt.scatter(embedding2[:, 0], embedding2[:, 1], c = y2, cmap='Spectral', alpha=0.7, s=s2)
plt.scatter(embedding2[:, 0], embedding2[:, 1], c = y2, cmap='Spectral')
# plt.savefig("humap_level2.svg")
plt.show()

y1 = hUmap.get_labels(1)
embedding1 = hUmap.get_embedding(1)
# plt.scatter(embedding1[:, 0], embedding1[:, 1], c = y1, cmap='Spectral', alpha=0.7, s=s1)
plt.scatter(embedding1[:, 0], embedding1[:, 1], c = y1, cmap='Spectral')
# plt.savefig("humap_level1.svg")
plt.show()


embedding0 = hUmap.get_embedding(0)
# plt.scatter(embedding0[:, 0], embedding0[:, 1], c = y, cmap='Spectral', alpha=0.7, s=s0)
plt.scatter(embedding0[:, 0], embedding0[:, 1], c = y, cmap='Spectral')
# plt.savefig("humap_level0.svg")
# indices0 = hUmap.get_indices(0)
# plt.scatter(embedding0[indices0, 0], embedding0[indices0, 1], c ='red', alpha=1, s=1)
plt.show()


print("Num points in scale %d: %d" % (2, len(embedding2)))
print("Num points in scale %d: %d" % (1, len(embedding1)))
print("Num points in scale %d: %d" % (0, len(embedding0)))

# sigmas0 = hUmap.get_sigmas(0)
# sigmas0 = np.sort(sigmas0)
# print("sigmas0: ")
# print(sigmas0[:10])
# df0 = pd.DataFrame({
# 	'sigmas': sigmas0
# 	})
# #ax = df0.plot.hist(bins=20, alpha=0.5)
# ax = sns.distplot(sigmas0)
# plt.show()


# sigmas1 = hUmap.get_sigmas(1)
# sigmas1 = np.sort(sigmas1)
# print("sigmas1: ")
# print(sigmas1[:10])
# df1 = pd.DataFrame({
# 	'sigmas': sigmas1
# 	})
# #ax = df1.plot.hist(bins=20, alpha=0.5)
# ax = sns.distplot(sigmas1)
# plt.show()






values =  hUmap.project(2, np.array([5, 7, 9]))
labels = hUmap.get_labels_selected()
influence = hUmap.get_influence_selected()
indices_selected = hUmap.get_indices_selected()
high_selected = embedding1[indices_selected]
print("selected:")
print(indices_selected.shape, high_selected.shape, values.shape)
s = transform_sizes(influence, 1, maxValue, rightMin=8, rightMax=300)
# plt.scatter(values[:, 0], values[:, 1], c = labels, cmap='Spectral',  alpha=0.7, s = s)
plt.scatter(values[:, 0], values[:, 1], c = labels, cmap='Spectral',  alpha=0.7, s = s)
# plt.savefig("humap_drilling_down.svg")
plt.show()

# values =  hUmap.project(1, np.array([5, 7, 9]))
# labels = hUmap.get_labels_selected()
# influence = hUmap.get_influence_selected()
# s = transform_sizes(influence, 1, maxValue, rightMin=8, rightMax=300)
# plt.scatter(values[:, 0], values[:, 1], c = labels, cmap='Spectral',  alpha=0.7, s = s)
# plt.show()


# cpp_umap = humap.UMAP("precomputed")
# start = time.time()
# cpp_umap.fit_hierarchy_sparse(data)
# end = time.time()
# print("time: %.5fs" % (end-start))

print(demap.DEMaP(high_selected, values))
ks=30
npres = neighborhood_preservation(high_selected, values, Khigh=ks)
print(npres)
plt.plot(np.arange(ks), npres)
plt.show()
pd_drilling = pd.DataFrame({
    'humap': npres
})
# pd_drilling.to_csv("humap_drilling.csv", index=False)


print(demap.DEMaP(third_level, embedding2))
ks = 30
npres = neighborhood_preservation(third_level, embedding2, Khigh=ks)
plt.plot(np.arange(ks), npres)
plt.show()
pd_level2 = pd.DataFrame({
    'humap': npres
})
# pd_level2.to_csv("humap_level2.csv", index=False)


print(demap.DEMaP(second_level, embedding1))
ks = 30
npres = neighborhood_preservation(second_level, embedding1, Khigh=ks)
plt.plot(np.arange(ks), npres)
plt.show()
pd_level1 = pd.DataFrame({
    'humap': npres
})
# pd_level1.to_csv("humap_level1.csv", index=False)



# precision2, recall2 = NNP(third_level, embedding2, Khigh=30)
# plt.plot(precision2, recall2)
# plt.show()

# precision, recall = NNP(second_level, embedding1, Khigh=30)
# plt.plot(precision, recall)
# plt.show()
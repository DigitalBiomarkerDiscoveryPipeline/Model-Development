import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

'''
Function that gets data points and cluster number(centroids), returns coordinates
of cluster centers
Default values: number of runs on different centroid seeds = 10, max runs = 300
'''
def run_kmeans(data, centroids, n_init=10, max_iter=300):
    KM = KMeans(n_clusters = centroids, n_init=n_init, max_iter=max_iter)
    y_KM = KM.fit_predict(data)
    return KM.cluster_centers_

'''
Function that helps to determine how many clusters to use by using trials of K clusters
The idea is to find the cluster number that gives the maximum reduction in inertia
'''
def elbow_method(data, num_k, n_init=10, max_iter=300):
    inertia = []
    for i in range(1, num_k):
        KM = KMeans(
        n_clusters=i,
        n_init=n_init, max_iter=max_iter
        )
        KM.fit_predict(data)
        inertia.append(KM.inertia_)
    
    plt.plot(range(1, num_k), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


'''Generate random sample (write another method to get data later?), just to show an example'''
# Assume we get this from the pre-processed data?
data, y = make_blobs(n_samples = 400, centers = 6, cluster_std = 0.60, random_state = 0)

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
elbow_method(data, 10)
# print(run_kmeans(data, 6))


    
    
    
    
    
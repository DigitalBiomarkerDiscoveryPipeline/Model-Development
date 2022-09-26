from array import array
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture 

def mean_shift(centers, predict_data=None):
    """Function that perform mean shift clustering, can also predict values if predict_data is passed

    Parameters
    ----------
    centers : 2D array like
        centers of data to perform clustering on
    predict_data : 2D array like, optional
        data to be predicted by the clustering, by default None

    Returns
    -------
    cluster_centers, labels, num_features, predict
        cluster_centers: centers after clustering
        labels: labels of each point
        num_features: number of features seen during fit
        predict: predicted values by the clustering for predict_data

    Raises
    ------
    Exception
        raise exception when normal array (non 2D array) is passed in as predict data
    """
    ms = MeanShift()
    clustering = ms.fit(centers)
    cluster_centers = clustering.cluster_centers_
    labels = clustering.labels_
    num_features = clustering.n_features_in_
    if type(predict_data) == type(array) or type(np.array):
        try: predicted = clustering.predict(predict_data)
        except: raise Exception ('Use 2D array for predict_data')
    else:
        predicted = None
    return cluster_centers, labels, num_features, predicted

def perform_DBSCAN(data, eps, min_samples):
    """Perform DBSCAN algorithm on a given set of data

    Parameters
    ----------
    data : 2D array-like
        array of data of interest to perform DBSCAN
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        This is not a maximum bound on the distances of points within a cluster. 
        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    min_samples : int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
        This includes the point itself.

    Returns
    -------
    labels, num_features, core_sample_indices, components
        labels: Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
        num_features: Number of features seen during fit.
        core_sample_indices: Indices of core samples.
        components: Copy of each core sample found by training.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    num_features = clustering.n_features_in_
    core_sample_indices = clustering.core_sample_indices_
    components = clustering.components_
    return labels, num_features, core_sample_indices, components

def gaussian_mixture_model(data, num_components, num_random_state=0, predict_data=None):
    """Perform unsupervised learning with gaussian mixture model for a given data, and make prediction if needed

    Parameters
    ----------
    data : 2D array
        Array of data to be fitted with Gaussian Mixture Model
    num_components : int
        number of underlying Gaussian distributions
    num_random_state : int
        random seed for initialization, by default 0
    predict_data : 2D array, optional
        array of data to be predicted from the model, by default None

    Returns
    -------
    predicted
        predicted is the predicted data of data passed into the model, which is predict_data
    """
    GMM = GaussianMixture(n_components=num_components, random_state=num_random_state).fit(data)
    if type(predict_data) == type(array) or type(np.array):
        predicted = GMM.predict(predict_data)
    else: predicted = None
    return predicted

def hierarchical_clustering(data, n_clusters=2, linkage='ward', distance_threshold=None):
    """Function that performs hiearchical clustering and fit to an array of data

    Parameters
    ----------
    data : 2D array
        data to be fitted
    n_clusters : int, default=2
        number of clusters to find
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
        The algorithm will merge the pairs of cluster that minimize this criterion.
        
        'ward' minimizes the variance of the clusters being merged.
        'average' uses the average of the distances of each observation of the two sets.
        'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
        'single' uses the minimum of the distances between all observations of the two sets.
    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be merged. 
        If not None, n_clusters must be None and compute_full_tree must be True.

    Returns
    -------
    num_clusters : int
        The number of clusters found by the algorithm
    labels : ndarray of shape (n_samples)
        Cluster labels for each point.
    num_leaves : int
        Number of leaves in the hierarchical tree
    num_connected_components : int
        The estimated number of connected components in the graph
    num_features : int
        number of features seen during fit
    """
    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, distance_threshold=distance_threshold)
    model.fit(data)
    num_clusters = model.n_clusters_
    labels = model.labels_
    num_leaves = model.n_leaves_
    num_connected_components = model.n_connected_components_
    num_features = model.n_features_in_
    return num_clusters, labels, num_leaves, num_connected_components, num_features
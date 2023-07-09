import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from numpy import unique
from numpy import where

def birch_clustering(X, n_clusters, threshold=0.1):
  birch = Birch(n_clusters=n_clusters, threshold=threshold)
  predicted_labels = birch.fit_predict(X)
  clusters = unique(predicted_labels)
  point_count_by_cluster = __find_point_count_by_cluster(clusters, predicted_labels, X)
  return point_count_by_cluster, predicted_labels


def dbscan_clustering(X, eps, min_samples):
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  predicted_labels = dbscan.fit_predict(X)
  clusters = unique(predicted_labels)
  point_count_by_cluster = __find_point_count_by_cluster(clusters, predicted_labels, X)
  return point_count_by_cluster, predicted_labels


def plot_clusters(X, labels, clusters):
  for cluster in clusters:
    row_ix = where(labels == cluster)
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
  plt.show()


def get_samples_from_clusters(num_samples_per_cluster, clusters_to_get_samples, X, predicted_cluster_labels):
  samples = []
  for cluster in clusters_to_get_samples:
    cluster_samples = X[predicted_cluster_labels == cluster]
    sampled_cluster_samples = cluster_samples[:num_samples_per_cluster]
    samples.extend(sampled_cluster_samples)
  samples = np.array(samples)
  return samples


def get_cluster_min_max_features(cluster, X, predicted_cluster_labels):
  cluster_data_points = X[predicted_cluster_labels == cluster]
  cluster_min_features = np.min(cluster_data_points, axis=0)
  cluster_max_features = np.max(cluster_data_points, axis=0)
  return cluster_min_features, cluster_max_features


def get_cluster_ranges(clusters, X, predicted_cluster_labels):
  ranges = {}
  for cluster in clusters:
    cluster_min_features, cluster_max_features = get_cluster_min_max_features(cluster, X, predicted_cluster_labels)
    ranges[cluster] = [[cluster_min_features[i], cluster_max_features[i]] for i in range(len(cluster_max_features))]
  return ranges


def __find_point_count_by_cluster(clusters, predicted_labels, X):
  point_count_by_cluster = {}
  for cluster in clusters: 
    row_ix = where(predicted_labels == cluster)
    cluster_points = X[row_ix]
    point_count_by_cluster[cluster] = len(cluster_points)
  return point_count_by_cluster
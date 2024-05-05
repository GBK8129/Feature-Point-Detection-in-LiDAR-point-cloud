import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import re
import threading

def extract_number(string):
    match = re.search(r'\d+$', string)
    if match:
        return int(match.group())
    else:
        return None

def calculate_normals(points, k=10):
    """
    tree = KDTree(points)
    normals = []

    for point in points:
        _, indices = tree.query([point], k=k)
        neighbors = points[indices[0]]

        # Calculate the covariance matrix
        covariance_matrix = np.cov(neighbors, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Get the normal vector associated with the smallest eigenvalue
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        normals.append(normal)

    return np.array(normals)"""

    # Build the KDTree outside the loop for efficiency
    tree = KDTree(points)

    normals = []

    # Perform nearest neighbor queries for all points at once
    _, indices = tree.query(points, k=k)
    for i, point_indices in enumerate(indices):
        neighbors = points[point_indices]

        # Calculate the covariance matrix
        covariance_matrix = np.cov(neighbors, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Get the normal vector associated with the smallest eigenvalue
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        normals.append(normal)

    return np.array(normals)


def derive_threshold(points, min_samples=5, eps=0.1, percentile=100):
    # Perform DBSCAN clustering on normal vectors
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Calculate average distance within clusters
    dot_prod = 0.6
    distances = []
    for label in np.unique(labels):
        cluster_points = points[labels == label]
        if len(cluster_points) > 1:
            nbrs = NearestNeighbors(n_neighbors=2).fit(cluster_points)
            distances.extend(nbrs.kneighbors(cluster_points)[0][:, 1])



    # Derive threshold based on percentiles of distances
    percentile_threshold = np.percentile(distances, percentile)
    return percentile_threshold

def feature_threads(csv_file):
    lidar_data = pd.read_csv(csv_file)
    lidar_points = lidar_data[['x', 'y', 'z']].values
    lidar_normals = calculate_normals(lidar_points)
    print(lidar_normals[0])
    threshold = derive_threshold(lidar_normals)
    if threshold>0.6:
      threshold=0.5999
    feature_indices = []
    num_feature_normals = 0

    for i, point in enumerate(lidar_points):
        neighbor_normals = lidar_normals[i]

        # Calculate the dot product of the normal vectors
        dot_products = np.dot(neighbor_normals, lidar_normals.T)
        # Count the number of normals with a dot product below the threshold
        num_feature_normals = np.sum(dot_products < threshold)

    j = 0
    while j < len(dot_products):
        if dot_products[j] < threshold:
            feature_indices.append(j)
        j += 1

    feature_points = lidar_points[feature_indices]

    name = Path(csv_file).name
    n = extract_number(name)
    w = name[:-1]
    if n and w:
        output_file = n + w + '_thr_' + str(threshold) + '.csv'
    elif w:
        output_file = w + '_thr_' + str(threshold) + '.csv'
    elif n:
        output_file = n + '_thr_' + str(threshold) + '.csv'
    else:
        output_file = '_thr_' + str(threshold) + '.csv'
    print(output_file)

    pd.DataFrame(feature_points, columns=['x', 'y', 'z']).to_csv(output_file, index=False)
    return num_feature_normals

if __name__ == '__main__':
    csv_files = ["data_bunny.csv"]#,"rf14s.csv"]#,"rf15s.csv"]#"drill2.csv"]#, "model_bunny.csv"]
    threads = []
    all_normals=[]
    for file in csv_files:
        thread = threading.Thread(target=feature_threads, args=(file,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed.")

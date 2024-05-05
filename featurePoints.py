import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# Define a function to calculate the normal vectors for each point
def calculate_normals(points, k=10):
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

    return np.array(normals)

def feature_point_calc(threshold):

    lidar_data = pd.read_csv('pavin2.csv')
    lidar_points = lidar_data[['x', 'y', 'z']].values
    lidar_normals = calculate_normals(lidar_points)
    
    feature_indices = []
    dot_products = []
    num_feature_normals=0
    
    # Iterate through the LiDAR points and check for feature points
    for i, point in enumerate(lidar_points):
        neighbor_normals = lidar_normals[i]
    
        # Calculate the dot product of the normal vectors
        dot_products = np.dot(neighbor_normals, lidar_normals.T)
        # Count the number of normals with a dot product below the threshold
        num_feature_normals = np.sum(dot_products < threshold)
    
    #print(num_feature_normals)
    
    j=0
    while(j<len(dot_products)):
        if(dot_products[j]<threshold):
            feature_indices.append(j)
        j=j+1
    

    feature_points = lidar_points[feature_indices]
    print(len(feature_points))
    
    pd.DataFrame(feature_points, columns=['x', 'y', 'z']).to_csv('model.csv', index=False)
    return num_feature_normals
    
if __name__ == "__main__":
   feature_point_calc(0.69)

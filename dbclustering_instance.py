import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Sample data: 3 vertices (rows), 3 frames (columns)
# Each vertex has its (x, y) positions and instance IDs in each frame.
# Here, the matrix contains rows as vertices and columns as frames.
matrix = np.array([
    [1, 1, 2],  # Vertex 1: Same instance across all frames (instance ID = 1)
    [1, 1, 2],  # Vertex 2: Same instance in frames 1 & 2 (instance ID = 2), switches to 3 in frame 3
    [1, 1, 2],  # Vertex 3: Same instance in frames 1 & 2 (instance ID = 3), switches to 2 in frame 3

])

# Assume we have some spatial information for each vertex in 2D space (for simplicity).
# In reality, you would extract these from your input data.
positions = np.array([
    [0, 0],  # Vertex 1 position (x1, y1)
    [1, 0],  # Vertex 2 position (x2, y2)
    [0, 1],  # Vertex 3 position (x3, y3)
])

# Define the weights for spatial distance and instance ID difference
alpha = 1  # weight for spatial distance
beta = 5   # weight for instance ID difference

# Function to calculate the distance matrix based on the custom distance metric
def custom_distance(positions, matrix, alpha=1, beta=1):
    num_vertices = positions.shape[0]
    dist_matrix = np.zeros((num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in range(num_vertices):
            # Spatial distance (Euclidean distance between positions)
            spatial_dist = np.linalg.norm(positions[i] - positions[j])
            
            # Instance ID similarity (binary similarity)
            instance_diff = np.sum(matrix[i] != matrix[j])  # Count of instance ID mismatches
            
            # Combine spatial distance and instance ID difference
            dist_matrix[i, j] = alpha * spatial_dist + beta * instance_diff
    
    return dist_matrix

# Calculate the distance matrix using the custom distance metric
distance_matrix = custom_distance(positions, matrix, alpha=alpha, beta=beta)

# Run DBSCAN on the distance matrix
dbscan = DBSCAN(eps=1, min_samples=2, metric='precomputed')
dbscan.fit(distance_matrix)

# Output the final instance assignments (cluster labels)
print(f"Instance IDs: {dbscan.labels_}")

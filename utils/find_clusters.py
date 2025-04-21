import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from mpl_toolkits.mplot3d import Axes3D
import trimesh

def compute_mahalanobis_distances(points):
    cov_matrix = np.cov(points, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean = np.mean(points, axis=0)
    return np.array([mahalanobis(p, mean, inv_cov_matrix) for p in points]).reshape(-1, 1)

def compute_pca_eigenvalue_ratios(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = pca.explained_variance_
    
    lambda1, lambda2, lambda3 = sorted(eigenvalues, reverse=True)
    ratio1 = lambda1 / lambda2 if lambda2 > 0 else 0
    ratio2 = lambda2 / lambda3 if lambda3 > 0 else 0
    
    return np.column_stack((np.full((points.shape[0], 1), ratio1), np.full((points.shape[0], 1), ratio2)))

def find_best_kmeans_clusters(points, k_min=2, k_max=2):
    best_k = k_min
    best_score = -1
    best_labels = None

    # Standardize the data
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Apply PCA to reduce dimensionality to 3D
    pca = PCA(n_components=3)
    points_pca = pca.fit_transform(points_scaled)

    # Compute Mahalanobis distance as an additional feature
    mahal_distances = compute_mahalanobis_distances(points_pca)

    # Compute PCA eigenvalue ratios
    pca_ratios = compute_pca_eigenvalue_ratios(points_pca)

    # Combine features
    points_augmented = np.hstack((points_pca, mahal_distances, pca_ratios))

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points_augmented)
        score = silhouette_score(points_augmented, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_k, best_labels

if __name__ == "__main__":
    # Generate a dummy 3D point cloud
    import matplotlib.pyplot as plt

    # Load points from the OBJ file

    mesh = trimesh.load('output/two_books_scan/objects/book_3_338_vertices.obj')
    points = np.array(mesh.vertices)

    # Find the best number of clusters and their labels
    best_k, best_labels = find_best_kmeans_clusters(points)

    print(f"Optimal number of clusters: {best_k}")

    # Visualize the clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=best_labels, cmap='viridis')
    ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    plt.title("3D Point Cloud Clustering")
    plt.show()

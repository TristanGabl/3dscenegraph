import open3d as o3d
import numpy as np

# Create a sample point cloud
pcd = o3d.geometry.PointCloud()
points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
pcd.points = o3d.utility.Vector3dVector(points)

# Create a 3D text annotation (manually as a sphere or custom marker)
text = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
text.translate([1, 1, 1])

# Visualizer setup
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add geometry
vis.add_geometry(pcd)
vis.add_geometry(text)

# You can add more markers for a legend here

# Run the visualizer
vis.run()
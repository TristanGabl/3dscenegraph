import open3d as o3d
import json
import numpy as np

# Load mesh with labels
scan = "ScanNet/scans/scene0134_02/scene0134_02"

mesh = o3d.io.read_triangle_mesh(scan + "_vh_clean_2.labels.ply")
mesh.compute_vertex_normals()

# Load segmentation and annotation data
with open(scan + ".aggregation.json") as f:
    annotations = json.load(f)
    pass

# Extract label positions




# Visualize mesh with text labels
o3d.visualization.draw_geometries([mesh])

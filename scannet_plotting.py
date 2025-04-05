import open3d as o3d
import json
import numpy as np

# Load mesh with labels
scan = "scannet_scans/scans/scene0134_02/scene0134_02"

mesh = o3d.io.read_triangle_mesh(scan + "_vh_clean_2.labels.ply")
mesh.compute_vertex_normals()

# Load segmentation and annotation data
with open(scan + ".aggregation.json") as f:
    annotations = json.load(f)

# Extract label positions
text_labels = []
for obj in annotations["segGroups"]:
    label = obj["label"]
    vertex_indices = obj["segments"]  # Segment indices
    if not vertex_indices:
        continue

    # Estimate object position (average vertex location)
    vertices = np.asarray(mesh.vertices)
    obj_center = np.mean(vertices[vertex_indices], axis=0)

    # Create a marker for the label position
    text_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=obj_center)
    text_labels.append(text_marker)
    print(f"Label: {label}, Position: {obj_center}")

# Visualize mesh with text labels
o3d.visualization.draw_geometries([mesh] + text_labels)

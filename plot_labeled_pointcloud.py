import numpy as np
import trimesh
import plotly.express as px
import pandas as pd

# Load the point cloud
name = "/teamspace/studios/this_studio/3dscenegraph/output/small_living_room/plot/small_living_room_pointcloud_classes"
mesh_vertices = np.array(trimesh.load_mesh(name + '.obj').vertices)
labels = np.load(name + ".npy")

# Invert the x-axis
mesh_vertices[:, 0] = -mesh_vertices[:, 0]

# Convert to DataFrame for easier handling with plotly
df = pd.DataFrame(mesh_vertices, columns=['x', 'z', 'y'])
df['label'] = labels

# Create an interactive scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), 
                     hover_data={'x': True, 'y': True, 'z': True, 'label': True})

fig.update_layout(title=name)  
fig.update_scenes(aspectratio=dict(x=(df['x'].max() - df['x'].min()) / 2, 
                                   y=(df['y'].max() - df['y'].min()) / 2, 
                                   z=(df['z'].max() - df['z'].min()) / 2
                                   ),
                  xaxis_autorange=False, 
                  yaxis_autorange=False, 
                  zaxis_autorange=False,
                  xaxis_range=[df['x'].min(), df['x'].max()], 
                  yaxis_range=[df['y'].min(), df['y'].max()], 
                  zaxis_range=[df['z'].min(), df['z'].max()]
                  )

fig.update_traces(marker=dict(size=1.5))  # Adjust point size for visibility
fig.write_html("pointcloud.html")  # Save the plot as an HTML file

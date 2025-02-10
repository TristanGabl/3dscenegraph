import numpy as np
import trimesh
import plotly.express as px
import pandas as pd
# from matplotlib import cm

def plot_labeled_pointcloud(name, metadata=None):

    #if metadata is not None:
    id_to_class = {i: name for i, name in enumerate(metadata.stuff_classes)}
    id_to_class_color = {i: color for i, color in enumerate(metadata.stuff_colors)}

    # Load the point cloud
    mesh_vertices = np.array(trimesh.load_mesh(name + '.obj').vertices)
    ids = np.load(name + ".npy")

    # Invert the x-axis
    mesh_vertices[:, 0] = -mesh_vertices[:, 0]

    # Convert to DataFrame
    df = pd.DataFrame(mesh_vertices, columns=['x', 'z', 'y'])
    df['id'] = ids
    df['labels'] = df['id'].apply(lambda x: 'unknown' if x == -1 else id_to_class[x])
    df['color'] = df['id'].apply(lambda x: 'rgb(0,0,0)' if x == -1 else f'rgb({id_to_class_color[x][0]},{id_to_class_color[x][1]},{id_to_class_color[x][2]})')
    
    colors = {id_to_class[i]: f'rgb({id_to_class_color[i][0]},{id_to_class_color[i][1]},{id_to_class_color[i][2]})' for i, name in id_to_class.items()}
    colors['unknown'] = 'rgb(0,0,0)'

    # unique_labels = np.unique(ids)
    # cmap = cm.get_cmap('tab10')
    # colors = {str(label): f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for label, (r, g, b, _) in zip(unique_labels, cmap(np.linspace(0, 1, len(unique_labels))))}

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', 
                        color_discrete_map=colors,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True})

    fig.update_layout(title=name, legend=dict(itemsizing='constant'))  
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

    fig.update_traces(marker=dict(size=1.5))

    # Add a callback or logic to adjust marker size based on camera zoom
    return fig


if __name__ == '__main__':
    name = "/teamspace/studios/this_studio/3dscenegraph/output/small_living_room/plot/small_living_room_pointcloud_classes"
    fig = plot_labeled_pointcloud(name)
    fig.write_html("/teamspace/studios/this_studio/plot_labeled_pointcloud.html")
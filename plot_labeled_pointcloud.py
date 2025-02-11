import numpy as np
import trimesh
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# from matplotlib import cm

def plot_labeled_pointcloud(name, ids, vertices, edges, objects, ids_to_class, ids_to_class_color):

    # Invert the x-axis
    vertices[:, 0] = -vertices[:, 0]

    # Convert to DataFrame
    df = pd.DataFrame(vertices, columns=['x', 'z', 'y'])
    df['id'] = ids
    df['labels'] = df['id'].apply(lambda x: 'unknown' if x == -1 else ids_to_class[x])
    df['color'] = df['id'].apply(lambda x: 'rgb(0,0,0)' if x == -1 else f'rgb({ids_to_class_color[x][0]},{ids_to_class_color[x][1]},{ids_to_class_color[x][2]})')
    
    colors = {ids_to_class[i]: f'rgb({ids_to_class_color[i][0]},{ids_to_class_color[i][1]},{ids_to_class_color[i][2]})' for i, name in ids_to_class.items()}
    colors['unknown'] = 'rgb(0,0,0)'

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in edges:
        x0, y0, z0 = df.iloc[edge[0]][['x', 'y', 'z']]
        x1, y1, z1 = df.iloc[edge[1]][['x', 'y', 'z']]
        edge_x += [x0, x1, None]  # None to break the line between edges
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', 
                        color_discrete_map=colors,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True})

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), hoverinfo='none')
    edge_trace.name = 'edges'
    fig.add_trace(edge_trace)

    # add objects
    for obj in objects:
        obj_vertices = [obj.x, obj.z, obj.y]
    
        obj_df = pd.DataFrame(obj_vertices, columns=['x', 'z', 'y'])
        obj_df['id'] = obj['ids']
        obj_df['labels'] = obj_df['id'].apply(lambda x: 'unknown' if x == -1 else ids_to_class[x])
        obj_df['color'] = obj_df['id'].apply(lambda x: 'rgb(0,0,0)' if x == -1 else f'rgb({ids_to_class_color[x][0]},{ids_to_class_color[x][1]},{ids_to_class_color[x][2]})')
        obj_colors = {ids_to_class[i]: f'rgb({ids_to_class_color[i][0]},{ids_to_class_color[i][1]},{ids_to_class_color[i][2]})' for i, name in ids_to_class.items()}
        obj_colors['unknown'] = 'rgb(0,0,0)'

        obj_trace = px.scatter_3d(obj_df, x='x', y='y', z='z', color='labels', 
                        color_discrete_map=obj_colors,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True})
        obj_trace.update_traces(marker=dict(size=3.5))
        fig.add_trace(obj_trace.data[0])


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

    fig.update_traces(marker=dict(size=3.5))

    # Add a callback or logic to adjust marker size based on camera zoom
    return fig


if __name__ == '__main__':
    name = "/teamspace/studios/this_studio/3dscenegraph/output/small_living_room/plot/small_living_room_pointcloud_classes"
    fig = plot_labeled_pointcloud(name)
    fig.write_html("/teamspace/studios/this_studio/plot_labeled_pointcloud.html")
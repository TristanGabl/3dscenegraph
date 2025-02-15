import numpy as np
import trimesh
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# from matplotlib import cm

def plot_labeled_pointcloud(name, ids, vertices, edges, objects, ids_to_class, ids_to_class_color):

    # Invert the x-axis
    vertices[:, 0] = -vertices[:, 0]
    for obj in objects:
        obj.x = -obj.x

    # name to color
    dict_name_to_color = {ids_to_class[i]: f'rgb({ids_to_class_color[i][0]},{ids_to_class_color[i][1]},{ids_to_class_color[i][2]})' for i, name in ids_to_class.items()}
    dict_name_to_color['unknown'] = 'rgb(0,0,0)'

    # Convert to DataFrame
    df = pd.DataFrame(vertices, columns=['x', 'z', 'y'])
    df['id'] = ids
    df['labels'] = df['id'].apply(lambda x: 'unknown' if x == -1 else ids_to_class[x])
    df['color'] = df['id'].apply(lambda x: 'rgb(0,0,0)' if x == -1 else f'rgb({ids_to_class_color[x][0]},{ids_to_class_color[x][1]},{ids_to_class_color[x][2]})')
    
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
                        color_discrete_map=dict_name_to_color,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True}
                        )

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), hoverinfo='none')
    edge_trace.name = 'edges'
    fig.add_trace(edge_trace)

    # add objects
    objects_for_df = [[obj.name + '_center',
                       obj.x, obj.y, obj.z,
                       ] 
                       for obj in objects]
    obj_df = pd.DataFrame(objects_for_df, columns=['name', 'x', 'z', 'y'])
    
    # change the color of the objects to magenta
    dict_name_to_magenta = {obj.name + '_center': f'rgb(255,0,255)' for obj in objects}

    obj_points = px.scatter_3d(obj_df, x='x', y='y', z='z', color='name',
                    color_discrete_map=dict_name_to_magenta,
                    hover_data={'x': True, 'y': True, 'z': True, 'name': True},
                    )
    
    for trace in obj_points.data:
        fig.add_trace(trace)

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
    
    return fig


if __name__ == '__main__':
    name = "/teamspace/studios/this_studio/3dscenegraph/output/small_living_room/plot/small_living_room_pointcloud_classes"
    fig = plot_labeled_pointcloud(name)
    fig.write_html("/teamspace/studios/this_studio/plot_labeled_pointcloud.html")
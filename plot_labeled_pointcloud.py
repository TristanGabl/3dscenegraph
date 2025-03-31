import numpy as np
import trimesh
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# from matplotlib import cm

def plot_labeled_pointcloud(self, name, ids, vertices, edges, edge_relationships, objects, ids_to_class, ids_to_class_color):

    # Invert the x-axis and switch the y and z axes
    vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    vertices[:, 0] = -vertices[:, 0]
    for obj in objects:
        obj.x = -obj.x
        obj.y, obj.z = obj.z, obj.y

    # name to color
    # dict_name_to_color = {ids_to_class[i]: f'rgb({ids_to_class_color[i][0]},{ids_to_class_color[i][1]},{ids_to_class_color[i][2]})' for i, name in ids_to_class.items()}
    # dict_name_to_color['unknown'] = 'rgb(0,0,0)'

    # object to color
    num_objects = len(objects)
    colors = [f'rgb({int(255 * i / (num_objects - 1))},{int(255 * (1 - i / (num_objects - 1)))},128)' for i in range(num_objects)]
    dict_object_to_color = {obj.name: colors[i] for i, obj in enumerate(objects)}
    dict_object_to_color['background'] = 'rgb(0,0,0)'
    
    # id to object
    dict_id_to_object = {}
    for obj in objects:
        if hasattr(obj, 'index_set'):
            for id_ in obj.index_set:
                dict_id_to_object[id_] = obj.name
        else:
            print(f"Warning: Object {obj.name} is missing 'index_set' attribute.")



    # id to color
    # dict_id_to_color = {id_: dict_object_to_color[dict_id_to_object[id_]] for id_ in dict_id_to_object}

    # add vertices
    df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    df['id'] = ids
    df['labels'] = [dict_id_to_object[i] if i in dict_id_to_object else 'background' for i,x in enumerate(ids)]
    df['color'] = [dict_object_to_color[dict_id_to_object[i]] if i in dict_id_to_object else 'rgb(0,0,0)' for i in ids]


    # df['color'] = df['id'].apply(lambda x: 'rgb(0,0,0)' if x == -1 else f'rgb({ids_to_class_color[x][0]},{ids_to_class_color[x][1]},{ids_to_class_color[x][2]})')
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', 
                        color_discrete_map=dict_object_to_color,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True}
                        )
    
    # add edges
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in edges:
        x0, y0, z0 = df.iloc[edge[0]][['x', 'y', 'z']]
        x1, y1, z1 = df.iloc[edge[1]][['x', 'y', 'z']]
        edge_x += [x0, x1, None]  # None to break the line between edges
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), hoverinfo='none')
    edge_trace.name = 'edges'
    fig.add_trace(edge_trace)

    # add objects
    objects_for_df = [[obj.name + ' center',
                       obj.x, obj.y, obj.z,
                       ] 
                       for obj in objects]
    obj_df = pd.DataFrame(objects_for_df, columns=['name', 'x', 'y', 'z'])
    
    dict_name_to_magenta = {obj.name + ' center': f'rgb(255,0,255)' for obj in objects}

    obj_points = px.scatter_3d(obj_df, x='x', y='y', z='z', color='name',
                    color_discrete_map=dict_name_to_magenta,
                    hover_data={'x': True, 'y': True, 'z': True, 'name': True},
                    )
    
    for trace in obj_points.data:
        fig.add_trace(trace)

    
    # add relationships edges between object centers
    edge_x = []
    edge_y = []
    edge_z = []
    relationships = []
    for obj in objects:
        for neighbor in obj.neighbors:
            x0, y0, z0 = obj.x, obj.y, obj.z
            x1, y1, z1 = objects[neighbor].x, objects[neighbor].y, objects[neighbor].z
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
            tmp = edge_relationships[obj.object_id][objects[neighbor].object_id]
            relationships += [tmp, tmp, tmp]
            
    

    
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='red', width=4), hoverinfo='text', text=relationships)
    edge_trace.name = 'relationships'
    fig.add_trace(edge_trace)


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
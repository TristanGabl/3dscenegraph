import argparse
import glob
import multiprocessing as mp
import os

import pickle
import re
import trimesh
import json

import cv2
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helper_repos', 'Mask2Former'))
from mask2former import add_maskformer2_config
from utils.scenegraph3d_objects import Objects

from utils.setup_logger import setup_logger
from plot.plot_labeled_pointcloud import plot_labeled_pointcloud
from dataset.relation_net import generate_edge_relationship, load_model

from utils.open_clip_ import compute_similarity
from utils.find_clusters import find_best_kmeans_clusters

class SceneGraph3D:
    def __init__(
        self,
        args, 
        DEBUG: bool,
        SAVE_VISUALIZATION: bool,
        SAVE_OBJECTS: bool,
        FORCE_MASK2FORMER: bool,
        SHORTCUT_0: bool,
        SHORTCUT_1: bool,
        USE_LLM: bool
    ):
        mp.set_start_method("spawn", force=True)
        self.logger = setup_logger(DEBUG)
        self.DEBUG = DEBUG
        self.logger.info("SceneGraph3D logger initialized")
        self.args = args
        self.logger.info("Arguments: " + str(args))
        self.SAVE_VISUALIZATION = SAVE_VISUALIZATION
        self.logger.info("SAVE_VISUALIZATION: " + str(SAVE_VISUALIZATION))
        self.SAVE_OBJECTS = SAVE_OBJECTS
        self.logger.info("SAVE_OBJECTS: " + str(SAVE_OBJECTS))
        self.FORCE_MASK2FORMER = FORCE_MASK2FORMER
        self.logger.info("FORCE_MASK2FORMER: " + str(FORCE_MASK2FORMER))
        self.SHORTCUT_0 = SHORTCUT_0
        self.logger.info("SHORTCUT_0: " + str(SHORTCUT_0))
        self.SHORTCUT_1 = SHORTCUT_1
        self.logger.info("SHORTCUT_1: " + str(SHORTCUT_1))
        self.USE_LLM = USE_LLM
        self.logger.info("USE_LLM: " + str(USE_LLM))

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.logger.info("Device: " + str(self.device))
        args.opts += ["MODEL.DEVICE", str(self.device)]
        self.config = self.setup_config(args)

        self.mask2former_predictor = DefaultPredictor(self.config)
        self.metadata = self.mask2former_predictor.metadata
        self.input_frames, self.input_scan_path = self.generate_input_frames()
        self.input_folder_name = self.input_scan_path.split('/')[-2] if self.input_scan_path.endswith('/') else self.input_scan_path.split('/')[-1]
        output_scan_path = os.path.join(self.args.output, self.input_folder_name)
        self.logger.info("Output path: " + output_scan_path)
        self.full_output_scan_path = os.path.join(output_scan_path, 'full')
        self.plot_output_scan_path = os.path.join(output_scan_path, 'plot')
        self.result_output_scan_path = os.path.join(output_scan_path, 'result')
        self.object_output_scan_path = os.path.join(output_scan_path, 'objects')
        self.number_input_image_paths = len(self.input_frames)
        self.logger.info("Number image frames: " + str(len(self.input_frames)))

        # check if scan has depth images
        if not any("depth" in file and file.endswith(".png") for file in os.listdir(self.input_scan_path)):
            self.logger.error("No depth images found in input directory, skipping depth map projection")

        # check if it is a scannet scan or 3d scanner app scan
        # self.scan_type = "scannet" if 
        if re.match(r"scene\d{4}_\d{2}", self.input_folder_name):
            self.scan_type = "scannet"
        else:
            self.scan_type = "3dscannerapp"
        self.logger.info("Scan type: " + self.scan_type)
            
    
    def generate_3d_scene_graph(self): # main function
        self.run_mask2former() # first use Mask2Former to get the panoptic segmentations
        # get all classes by iterating over all segment_info dictionaries in the processed frames
        self.classes = np.unique([segment_info['category_id'] for path in self.processed_frame_paths for segment_info in pickle.load(open(path + '.pkl', 'rb'))[1]])
    
        # load the mesh vertices and faces
        self.mesh_vertices = np.array(trimesh.load_mesh(os.path.join(self.input_scan_path, 'export_refined.obj')).vertices)
        self.mesh_faces = np.array(trimesh.load_mesh(os.path.join(self.input_scan_path, 'export_refined.obj')).faces)

        # distribute the panoptic segmentations from the images to the mesh vertices and remember in which frames a vertex was observed
        self.mesh_vertices_votes_global, self.mesh_vertices_frame_observations = self.distribute_panoptic_segmentations()

        # fuse votes to the vote that is most frequent (global), except for 0 
        self.mesh_vertices_classes = np.apply_along_axis(lambda row: self.classes[np.argmax(row)] if np.any(row) else -1, 1, self.mesh_vertices_votes_global)

        # extract class names and colors from metadata
        self.id_to_class = {i: name for i, name in enumerate(self.metadata.stuff_classes)}
        self.id_to_class[-1] = "background"
        self.id_to_class_color = {i: color for i, color in enumerate(self.metadata.stuff_colors)}
        self.id_to_class_color[-1] = [0, 0, 0] # black

        with open("label_mapping/coco_id_to_name.json", "w") as json_file:
            json.dump(self.id_to_class, json_file, indent=4)


        # create connected graph from the mesh vertices 
        self.mesh_edges = self.create_graph_edges()

        # traverse the graph to get Objects()
        # They are of type Objects(name, index_set, center, relations)
        self.edges_boarders = self.mesh_edges[np.logical_and(self.mesh_vertices_classes[self.mesh_edges[:, 0]] != self.mesh_vertices_classes[self.mesh_edges[:, 1]], 
                                              np.logical_and(self.mesh_vertices_classes[self.mesh_edges[:, 0]] != -1, self.mesh_vertices_classes[self.mesh_edges[:, 1]] != -1))]
        
        self.objects = self.create_3dscenegraph_objects()

        # make extra check in case two same objects are next to each other
        # self.objects = self.duplicate_double_check_mask2former(self.objects)
        # self.objects = self.duplicate_double_check_kmeans(objects)
        self.update_neighbors(self.objects, self.edges_boarders)

        # assign lost vertices to nearest object by using BFS
        # self.objects = self.assign_lost_vertices_to_nearest_object(objects)

        if self.SAVE_OBJECTS:
            self.save_object_vertices(self.objects)
        
        # save objects into a json file
        if not os.path.exists(self.result_output_scan_path):
            os.makedirs(self.result_output_scan_path, exist_ok=True)
        self.objects_json = [{k: v for k, v in obj.__dict__.items() if k != 'index_set' and v != ''} for obj in self.objects]
        with open(os.path.join(self.result_output_scan_path, 'objects.json'), 'w') as f:
            json.dump(self.objects_json, f, indent=4)
        
        coco_name_to_name_simplified = json.load(open("label_mapping/coco_name_to_name_simplified.json", 'r'))
        relation_net_model, name2idx, label2idx, idx2label = load_model("dataset/relation_model.pth", self.device)
        self.edge_relationships = [["" for _ in range(len(self.objects))] for _ in range(len(self.objects))]
        for object1 in self.objects:
            for object2 in self.objects:
                if object1.object_id in set(object2.neighbors):
                    obj1 = object1.__class__(**object1.__dict__.copy())
                    obj2 = object2.__class__(**object2.__dict__.copy())
                    obj1.name = coco_name_to_name_simplified[obj1.name]
                    obj2.name = coco_name_to_name_simplified[obj2.name]
                    edge_forward = generate_edge_relationship(obj1, obj2, relation_net_model, name2idx, label2idx, idx2label)
                    edge_backward = generate_edge_relationship(obj2, obj1, relation_net_model, name2idx, label2idx, idx2label)
                    self.edge_relationships[object1.object_id][object2.object_id] = edge_forward
                    self.edge_relationships[object2.object_id][object1.object_id] = edge_backward                    
    
        self.logger.info("Finished generating edge relationships")
       
        # plot everything
        self.save_segmented_pointcloud()

        
    def run_mask2former(self):
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images"
        )

        self.processed_frame_paths = [] # saved with no suffix
        for frame in self.input_frames:
            frame_path = os.path.join(self.input_scan_path, frame)
            image_path = frame_path + '.jpg'
            image_info_path = frame_path + '.json'
            output_image_path = os.path.join(self.full_output_scan_path, frame)
            inference_output_path = output_image_path + '.pkl'

            image = read_image(image_path, format="BGR")

            if not os.path.exists(inference_output_path) or self.FORCE_MASK2FORMER:
                pbar.set_description("Running Mask2Former on image: {}".format(frame))
                pbar.update()

                image_info = json.load(open(image_info_path, 'r'))

                if self.scan_type == "scannet":
                    if not any(key in image_info.keys() for key in ["calibrationColorIntrinsic", "calibrationDepthIntrinsic", "Pose", "depthShift", "depthWidth", "depthHeight", "colorWidth", "colorHeight"]):
                        self.logger.warning("{}: Frame not in input frames, skipping this image".format(frame))
                        # remove frame
                        self.input_frames.remove(frame)
                        continue
                else: # "3dscannerapp"
                    if not any(key in image_info.keys() for key in ["cameraPoseARFrame", "projectionMatrix"]):
                        self.logger.warning("{}: None of cameraPoseARFrame, projectionMatrix, or mvp found in image info json file, skipping this image".format(frame))
                        # remove frame
                        self.input_frames.remove(frame)
                        continue

                

                os.makedirs(self.full_output_scan_path, exist_ok=True)

                # run inference
                predictions = self.mask2former_predictor(image)

                # BGR to RGB
                image = image[:, :, ::-1]

                # save inference output
                with open(inference_output_path, 'wb') as f:
                    pickle.dump((predictions["panoptic_seg"][0].to(torch.device("cpu")), predictions["panoptic_seg"][1]), f)
                
                # save image info
                if self.scan_type == "scannet":
                    image_info_relevant = {key: image_info[key] for key in ['calibrationColorIntrinsic', 'calibrationDepthIntrinsic', 'Pose', 'depthShift', 'depthWidth', 'depthHeight', 'colorWidth', 'colorHeight']}
                    with open(output_image_path + '.json', 'w') as f:
                        json.dump(image_info_relevant, f)
                else: # "3dscannerapp" 
                    image_info_relevant = {key: image_info[key] for key in ['cameraPoseARFrame', 'projectionMatrix']}
                    with open(output_image_path + '.json', 'w') as f:
                        json.dump(image_info_relevant, f)
        
                if self.SAVE_VISUALIZATION:
                    assert "panoptic_seg" in predictions
                    panoptic_seg, panoptic_seg_info = predictions["panoptic_seg"]
                    visualizer = Visualizer(image, 
                                        MetadataCatalog.get(self.config.DATASETS.TEST[0] if len(self.config.DATASETS.TEST) else "__unused"), 
                                        instance_mode=ColorMode.IMAGE
                                        )
                    vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), panoptic_seg_info)
                    vis_output.save(output_image_path + '.jpg')
                    self.logger.debug("Saved visualization to: {}".format(output_image_path + '.jpg'))
                    
            else:
                pbar.set_description("Skipping image (inference already saved): {}".format(frame))
                pbar.update()

            # if everything is successful, add the path to the list of processed images
            self.processed_frame_paths.append(output_image_path)
        pbar.close()

        self.logger.important("Finished running Mask2former on images")

    def duplicate_double_check_mask2former(self, objects):
        pbar = tqdm.tqdm(
            total=len(objects),
            unit="objects"
        )

        new_objects = []
        new_objects_id = 0  
        for obj in objects:
            # go over images that see the object and find image that features the most instances of the object type
            # frames_tmp = [frame for idx in blob for frame in self.mesh_vertices_frame_observations[idx]]
            frames_tmp = np.unique([frame for idx in obj.index_set for frame in self.mesh_vertices_frame_observations[idx]])

            best_perspective_frame = None
            object_count_best = -1
            for frame in frames_tmp:
                image_path = os.path.join(self.full_output_scan_path, frame)
                panoptic_seg, panoptic_seg_info = pickle.load(open(image_path + '.pkl', 'rb'))

                object_count = np.sum(segment_info['category_id'] == obj.class_id for segment_info in panoptic_seg_info)
                if object_count > object_count_best:
                    best_perspective_frame = frame
                    object_count_best = object_count

                    
            # load from saved files
            image_path = os.path.join(self.full_output_scan_path, best_perspective_frame)
            panoptic_seg, panoptic_seg_info = pickle.load(open(image_path + '.pkl', 'rb'))
            image_info = json.load(open(image_path + '.json', 'r'))

            local_class_values = [i for i, _ in enumerate(panoptic_seg_info) if panoptic_seg_info[i]['category_id'] == obj.class_id]
            if any(keyword in obj.name.lower().replace('-', ' ').split() for keyword in ["floor", "wall", "table", "ceiling"]) or len(local_class_values) == 1:
                print("Skipping floor or wall, etc., or single object")
                obj.object_id = new_objects_id
                obj.name = obj.name + " " + str(new_objects_id)
                new_objects_id += 1
                new_objects.append(obj)
                pbar.update()
                continue
            pbar.set_description("Checking for duplicates (mask2former): {}".format(obj.name))
            pbar.update()            

            # create mask for object
            pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
            projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
            view_matrix = np.linalg.inv(pose)
            mvp = np.dot(projection_matrix, view_matrix)

            projections = self.project_points_to_image(self.mesh_vertices[obj.index_set], mvp, panoptic_seg.shape[1], panoptic_seg.shape[0])
            projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
            projections_2d = projections_2d[
                (projections_2d[:, 0] >= 0) & 
                (projections_2d[:, 0] < panoptic_seg.shape[1]) &
                (projections_2d[:, 1] >= 0) & 
                (projections_2d[:, 1] < panoptic_seg.shape[0])
            ]

            # Check the value of each point in projections_2d in the panoptic_seg
            point_values = panoptic_seg[projections_2d[:, 1], projections_2d[:, 0]]
            
            # If there are multiple unique values for the object's class, split the object

            self.logger.info(f"Object {obj.name} has multiple segments in the panoptic segmentation.")
            for value in local_class_values:
                new_index_set = np.array(obj.index_set)[np.where(np.isin(point_values, value+1))[0]]
                new_object = Objects(
                    obj.name + f"_{new_objects_id}",
                    new_objects_id,
                    obj.class_id,
                    new_index_set,
                    np.mean(self.mesh_vertices[new_index_set], axis=0)[0],
                    np.mean(self.mesh_vertices[new_index_set], axis=0)[1],
                    np.mean(self.mesh_vertices[new_index_set], axis=0)[2],
                    [],
                    [],
                    best_perspective_frame
                )
                new_objects.append(new_object)
                new_objects_id += 1
        
        pbar.close()
        #update neighbors
        self.update_neighbors(new_objects, self.edges_boarders)
        return new_objects

    
    def assign_lost_vertices_to_nearest_object(self, objects):

        pass

            

    def duplicate_double_check_kmeans(self, objects):
        pbar = tqdm.tqdm(
            total=len(objects),
            unit="objects"
        )

        new_objects = []
        new_objects_id = 0  
        for obj in objects:
            if any(keyword in obj.name.lower().replace('-', ' ').split() for keyword in ["floor", "wall", "table", "ceiling"]):
                print("Skipping floor or wall")
                obj.object_id = new_objects_id
                obj.name = obj.name + " " + str(new_objects_id)
                new_objects_id += 1
                new_objects.append(obj)
                pbar.update()
                continue
            pbar.set_description("Checking for duplicates (kmeans): {}".format(obj.name))
            pbar.update()

            image_path = os.path.join(self.input_scan_path, obj.best_perspective_frame) + '.jpg'
            image = read_image(image_path, format="BGR")
            image_info_path = os.path.join(self.full_output_scan_path, obj.best_perspective_frame) + '.json'
            image_info = json.load(open(image_info_path, 'r'))

            # create mask for object
            pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
            projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
            view_matrix = np.linalg.inv(pose)
            mvp = np.dot(projection_matrix, view_matrix)

            projections = self.project_points_to_image(self.mesh_vertices[obj.index_set], mvp, image.shape[1], image.shape[0])
            projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
            projections_2d = projections_2d[
                (projections_2d[:, 0] >= 0) & 
                (projections_2d[:, 0] < image.shape[1]) & 
                (projections_2d[:, 1] >= 0) & 
                (projections_2d[:, 1] < image.shape[0])
            ]


            # create mask for object
            # mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # mask[projections_2d[:, 1], projections_2d[:, 0]] = 255

            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # mask[:mask.shape[0] // 2, :mask.shape[1] // 2] = 255

            # create convex hull of the projections
            hull = cv2.convexHull(projections_2d)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # apply mask to the image
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            cropped_masked_image = masked_image[
                np.min(projections_2d[:, 1]):np.max(projections_2d[:, 1]),
                np.min(projections_2d[:, 0]):np.max(projections_2d[:, 0])
            ]


            # compute similarity between the masked image and the original image
            save_image_path = os.path.join(self.full_output_scan_path, obj.best_perspective_frame) + '_double_check_ ' + obj.name + '.jpg'
            cv2.imwrite(save_image_path, cropped_masked_image)
            obj_name_no_numbers = ''.join([char for char in obj.name if not char.isdigit()])
            similarity_1 = compute_similarity(save_image_path, "There is only one " + obj_name_no_numbers)
            similarity_2 = compute_similarity(save_image_path, "There are multiple " + obj_name_no_numbers)
            self.logger.info("Similarity for one {}: {}".format(obj_name_no_numbers, similarity_1))
            self.logger.info("Similarity for multiple {}: {}".format(obj_name_no_numbers, similarity_2))
            if similarity_1 < similarity_2:
                # Use k-means clustering to find new objects
                number_new_cluster, new_index_sets = find_best_kmeans_clusters(self.mesh_vertices[obj.index_set])
                self.logger.info("Found {} clusters for object {}".format(number_new_cluster, obj.name))
                
                new_index_sets = [np.array(obj.index_set)[np.where(new_index_sets == i)[0]] for i in range(number_new_cluster)]
                
                # create new objects
                split_objects = []
                for i, new_index_set in enumerate(new_index_sets):
                    new_object = Objects(
                        obj.name + " " + str(new_objects_id),
                        new_objects_id,
                        obj.class_id,
                        new_index_set,
                        np.mean(self.mesh_vertices[new_index_set], axis=0)[0],
                        np.mean(self.mesh_vertices[new_index_set], axis=0)[1],
                        np.mean(self.mesh_vertices[new_index_set], axis=0)[2],
                        [],
                        [],
                        obj.best_perspective_frame
                    )
                    split_objects.append(new_object)
                    new_objects_id += 1
                
                
                new_objects.extend(split_objects)

            else:
                new_objects.append(obj)

        pbar.close()
        #update neighbors
        self.update_neighbors(new_objects, self.edges_boarders)
        return new_objects


    def distribute_panoptic_segmentations(self):
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images",
        )

        # initialize the mesh vertices votes matrix
        mesh_vertices_votes_global = np.zeros((self.mesh_vertices.shape[0], len(self.classes)), dtype=int)

        # initialize vertex observations
        mesh_vertices_frame_observations = [set() for _ in range(self.mesh_vertices.shape[0])]

        # iterate over the processed images and project the 3d points into the image and apply panoptic segmentation
        for frame in self.processed_frame_paths:
            pbar.set_description("Projecting points into image and applying panoptic segmentation: {}".format(os.path.basename(frame)))
            
            # load from saved files
            panoptic_seg, panoptic_seg_info = pickle.load(open(frame + '.pkl', 'rb'))
            image_info = json.load(open(frame + '.json', 'r'))

            # load the image and depth map
            image = cv2.imread(frame + '.jpg')
            depth_map_path = os.path.join(self.input_scan_path, frame.split('/')[-1].replace("frame", "depth") + ".png")
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED) / 1000 # convert to meters
            if self.scan_type == "scannet":
                # depth_map = depth_map
                depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # resize to image size
            else:
                depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # resize to image size

            # project the 3d point cloud into the image and filter out points that are not in the image
            if self.scan_type == "scannet":
                projections_filtered, projections_filtered_mask = self.project_pointcloud_scannet(image_info, image)
            else:
                projections_filtered, projections_filtered_mask = self.project_pointcloud_3dscannerapp(image_info, image)

            # remember what image a vertex was observed in
            [mesh_vertices_frame_observations[idx].add(frame.split("/")[-1]) for idx in np.where(projections_filtered_mask)[0]]
            
            # projections_filtered has shape (n, 2) where n is the number of points, depth_map has shape (h, w), i want to get the depth of each point
            if self.scan_type == "scannet":
                # K_color = np.array(image_info['calibrationColorIntrinsic'])[:3, :3]
                # K_depth = np.array(image_info['calibrationDepthIntrinsic'])[:3, :3]
                # uvz = projections_filtered  # shape (N, 3)
                # u_rgb, v_rgb, Z = uvz[:, 0], uvz[:, 1], uvz[:, 2]

                # # Backproject RGB pixels to 3D
                # x = (u_rgb - K_color[0, 2]) * Z / K_color[0, 0]
                # y = (v_rgb - K_color[1, 2]) * Z / K_color[1, 1]
                # xyz = np.stack([x, y, Z], axis=1)  # shape (N, 3)

                # # Project to depth image
                # xyz_h = xyz.T  # shape (3, N)
                # uv_depth = K_depth @ (xyz_h / xyz_h[2])  # shape (3, N)
                # projections_filtered_depth = uv_depth[:2].T  # shape (N, 2)
                # depth_array = depth_map[projections_filtered_depth[:, 1].astype(int), projections_filtered_depth[:, 0].astype(int)]
                depth_array = depth_map[projections_filtered[:, 1].astype(int), projections_filtered[:, 0].astype(int)]
            else:
                depth_array = depth_map[projections_filtered[:, 1].astype(int), projections_filtered[:, 0].astype(int)]


            # for debugging projected points
            if self.SAVE_VISUALIZATION and not self.SHORTCUT_0:
                img_projected = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # add grey border around the image for debugging
                border_offset = 50
                img_projected = cv2.copyMakeBorder(img_projected, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                for i, point in enumerate(projections_filtered[:,:2]):
                    if abs(projections_filtered[i, 2] - depth_array[i]) <= 0.03:
                        cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 2, (0, 0, 255), -1)
                    else:
                        cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 2, (228, 228, 228), -1)

                cv2.imwrite(frame + '_projections.jpg', img_projected) 
                self.logger.debug("saved projections to {}_projections.jpg".format(frame))
            

            # we use panoptic_seg to get votes for object classes for each 3d point
            local_class_ids = np.array([segment_info['id'] for segment_info in panoptic_seg_info])
            # create matrix that stores the votes for each class for each filtered point
            projections_class_votes_local = np.zeros((len(projections_filtered), len(local_class_ids)))

            # distribute the class votes to the mesh vertices
            projections_class_votes_local, mesh_vertices_votes_global = self.distribute_class_votes(mesh_vertices_votes_global, projections_filtered, projections_filtered_mask, panoptic_seg, panoptic_seg_info, depth_array, projections_class_votes_local)
            
            # fuse votes to the vote that is most frequent except for 0 (local)
            projections_class_votes_local = np.apply_along_axis(lambda row: local_class_ids[np.argmax(row)] if np.any(row) else -1, 1, projections_class_votes_local)
            
            # for debugging point classes
            if self.DEBUG:
                number_of_classes = len(local_class_ids)
                number_of_class_points = np.sum(projections_class_votes_local != 0)
                self.logger.debug("number_of_classes: {}".format(number_of_classes))
                self.logger.debug("number_of_class_points: {}".format(number_of_class_points))
                if self.SAVE_VISUALIZATION and not self.SHORTCUT_1:
                    image_fused_votes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                    # add grey border around the image for debugging
                    border_offset = 50
                    image_fused_votes = cv2.copyMakeBorder(image_fused_votes, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                    # Define a colormap for the classes
                    colors = np.zeros((len(projections_filtered), 3))
                    if number_of_classes > 0:
                        colormap = plt.cm.get_cmap("tab20", np.max(local_class_ids) + 1)
                    else:
                        colormap = None # no classes found

                    # Map each class to a color
                    for i, class_id in enumerate(local_class_ids):
                        if class_id == -1:
                            colors[projections_class_votes_local == class_id] = (0, 0, 0)
                        else:
                            colors[projections_class_votes_local == class_id] = colormap(class_id)[:3]
                    
                    for i, point in enumerate(projections_filtered[:,:2]): # leave out the depth
                        cv2.circle(image_fused_votes, tuple(point.ravel().astype(int) + border_offset), 4, (colors[i] * 255), -1)

                    cv2.imwrite(frame + '_fused_votes.jpg', image_fused_votes)
                    self.logger.debug("saved projections with votes to {}_fused_votes.jpg".format(frame))

            pbar.update()
        pbar.close()

        self.logger.important("Finished distributing panoptic segmentations to mesh vertices")
        return mesh_vertices_votes_global, mesh_vertices_frame_observations


    def distribute_class_votes(self, mesh_vertices_votes_global, projections_filtered, projections_filtered_mask, panoptic_seg, panoptic_seg_info, depth_array, projections_class_votes_local):
        for category_id_local, category_id_global in enumerate([seg["category_id"] for seg in panoptic_seg_info], start=1):
            # filter out for current local category
            mask = panoptic_seg == category_id_local 
            if mask.sum() == 0:
                continue
        
            # transform to a boolean mask for the filtered points
            mask = mask[projections_filtered[:, 1], projections_filtered[:, 0]]
            # add criteria for depth (+-0.05 meters)
            mask = mask & (np.abs(depth_array - projections_filtered[:, 2]) < 0.05)

            # add the votes to the global mesh vertices according to the number of classes in the panoptic segmentation 
            # (if there are more classes, the image has more of an overview of the scene -> better segmentation)
            weight_factor = len(panoptic_seg_info) * len(panoptic_seg_info)
            mesh_vertices_votes_global[projections_filtered_mask, np.where(self.classes == category_id_global)[0][0]] += mask.numpy() * weight_factor
            # add the votes to the local mesh projections
            projections_class_votes_local[:, category_id_local-1] = mask.numpy()
        
        return projections_class_votes_local, mesh_vertices_votes_global
        
    def project_pointcloud_scannet(self, image_info, image):
        pose = np.array(image_info['Pose']).reshape((4, 4))
        pose = np.linalg.inv(pose)
        # Convert 3D points to homogeneous coordinates
        points_homogeneous = np.hstack((self.mesh_vertices, np.ones((self.mesh_vertices.shape[0], 1))))  # Shape: [N, 4]
        points_camera = np.dot(points_homogeneous, pose.T)

        K = np.array(image_info['calibrationColorIntrinsic'])[:3, :3]


        # Project to image plane
        points_2d = (K @ points_camera[:, :3].T).T  # [N, 3]

        # Round and filter inside image bounds
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]
        points_2d[:, :2] = np.round(points_2d[:, :2]).astype(int) 

        projections_filtered_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0]) &
            (points_2d[:, 2] > 0)  # Ensure points are in front of the camera
        )

        # Filter points
        projections_filtered = points_2d[projections_filtered_mask]
        

        if self.DEBUG:
            self.logger.debug("number_points_projected: {}".format(len(projections_filtered)))

        return projections_filtered, projections_filtered_mask


        



    def project_pointcloud_3dscannerapp(self, image_info, image):
        # compute the model view projection matrix

        pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
        projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
        view_matrix = np.linalg.inv(pose)
        mvp = np.dot(projection_matrix, view_matrix)
        
        # project the 3d point cloud and filter out points that are not in the image
        projections = self.project_points_to_image(self.mesh_vertices, mvp, image.shape[1], image.shape[0])
        projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
        projections[:, :2] = projections_2d # store the z coordinate
        projections_filtered_mask = (projections[:, 0] >= 0) & (projections[:, 0] < image.shape[1]) & (projections[:, 1] >= 0) & (projections[:, 1] < image.shape[0])
        projections_filtered = projections[projections_filtered_mask]

        if self.DEBUG:
            self.logger.debug("number_points_projected: {}".format(len(projections_filtered)))
       # projections_filtered
        # array([[1.12100000e+03, 6.00000000e+00, 2.23226316e+00],
        #        [1.12700000e+03, 1.10000000e+01, 2.23858557e+00],
        #        [1.13200000e+03, 4.00000000e+00, 2.23749928e+00],
        #        ...,
        #        [1.77700000e+03, 4.24000000e+02, 1.62636298e+00],
        #        [1.78700000e+03, 4.30000000e+02, 1.60979556e+00],
        #        [1.79200000e+03, 4.34000000e+02, 1.60229302e+00]])

        return projections_filtered, projections_filtered_mask
    
    def create_graph_edges(self):
        edges = np.vstack([self.mesh_faces[:, [0, 1]], self.mesh_faces[:, [1, 2]], self.mesh_faces[:, [2, 0]]])
        
        # remove double edges and also reverse edges
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        # remove edges that connect two vertices with different classes
        # edges = edges[self.mesh_vertices_classes[edges[:, 0]] == self.mesh_vertices_classes[edges[:, 1]]]
        return edges
    
    def create_3dscenegraph_objects(self):
        # Define an undirected graph and find connected components
        G = nx.Graph()
        # remove edges that connect two vertices with different classes
        edges_single_classes = self.mesh_edges[self.mesh_vertices_classes[self.mesh_edges[:, 0]] == self.mesh_vertices_classes[self.mesh_edges[:, 1]]]
        G.add_edges_from(edges_single_classes)  # Add edges to the graph, also adds the vertices
        blobs = list(nx.connected_components(G))

        # remove small blobs and blobs corresponding to background
        blobs = [list(blob) for blob in blobs if np.all(self.mesh_vertices_classes[list(blob)] != -1) and len(blob) > 30]
         
        self.logger.info("Number of objects found in Graph: {}".format(len(blobs)))

        # create list of Objects() from the blobs
        objects = [] 
        for i, blob in enumerate(blobs):
            # use one vertex to get object name
            object_id = i
            object_class = self.id_to_class[self.mesh_vertices_classes[blob[0]]]
            # check for duplicates
            class_id = self.mesh_vertices_classes[blob[0]]
            center = np.mean(self.mesh_vertices[blob], axis=0)
            min_coords = np.min(self.mesh_vertices[blob], axis=0)
            max_coords = np.max(self.mesh_vertices[blob], axis=0)
            size_x = max_coords[0] - min_coords[0]
            size_y = max_coords[1] - min_coords[1]
            size_z = max_coords[2] - min_coords[2]

            # frames_tmp = [frame for idx in blob for frame in self.mesh_vertices_frame_observations[idx]]
            # best_perspective_frame = max(set(frames_tmp), key=frames_tmp.count)

            # Calculate the bounding box around the vertices of the object

            objects.append(Objects(name=object_class, 
                                   object_id=object_id, 
                                   class_id=class_id, 
                                   x=center[0], 
                                   y=center[1], 
                                   z=center[2],
                                   size_x=size_x,
                                   size_y=size_y,
                                   size_z=size_z,
                                   index_set=blob))

        # will be done after object duplicate check
        # self.update_neighbors(objects, self.edges_boarders)
            
        return objects
    
    def update_neighbors(self, objects, edges_borders):
        self.logger.info("Updating neighbors...")

        # create a mapping from vertex index to object id
        index_to_obj = {}
        for obj_id, obj in enumerate(objects):
            for idx in obj.index_set:
                index_to_obj[idx] = obj_id

        for a, b in edges_borders:
            object_id_0 = index_to_obj.get(a)
            object_id_1 = index_to_obj.get(b)
            if object_id_0 is None or object_id_1 is None:
                continue

            # add each other as neighbors
            if object_id_1 not in objects[object_id_0].neighbors:
                objects[object_id_0].neighbors.append(object_id_1)
            if object_id_0 not in objects[object_id_1].neighbors:
                objects[object_id_1].neighbors.append(object_id_0)
        self.logger.info("Updated neighbors!")


    def save_segmented_pointcloud(self):
        path_full = os.path.join(self.full_output_scan_path, self.input_folder_name)
        path_plot = os.path.join(self.plot_output_scan_path, self.input_folder_name)
        np.save(path_full + '_pointcloud_classes.npy', self.mesh_vertices_classes)

        os.makedirs(self.plot_output_scan_path, exist_ok=True)
        np.save(path_plot + '_pointcloud_classes.npy', self.mesh_vertices_classes)

        self.logger.info("saved pointcloud_classes.npy")

        # copy .obj file into output directories
        os.system("cp " + os.path.join(self.input_scan_path, 'export_refined.obj') + " " + path_full + '_pointcloud_classes.obj')
        os.system("cp " + os.path.join(self.input_scan_path, 'export_refined.obj') + " " + path_plot + '_pointcloud_classes.obj')
        self.logger.info("copied export_refined.obj")

        # save metadata as npy file
        # np.save(path_full + '_metadata.npy', self.metadata)
        # np.save(path_plot + '_metadata.npy', self.metadata)
        # self.logger.info("saved metadata.npy")
    
        if self.SAVE_VISUALIZATION:
            # Load the point cloud
            name = path_plot + '_pointcloud_classes'
            
            edges_single_classes = self.mesh_edges[self.mesh_vertices_classes[self.mesh_edges[:, 0]] == self.mesh_vertices_classes[self.mesh_edges[:, 1]]]
            fig = plot_labeled_pointcloud(self, name, self.mesh_vertices_classes, self.mesh_vertices, edges_single_classes, self.edge_relationships, self.objects, self.id_to_class, self.id_to_class_color)

            fig.write_html(name + '.html')
            fig.write_html(name + '.html')

            self.logger.info("saved pointcloud visualization html")

    def save_object_vertices(self, objects):
        # Save each object as a separate .obj file
        if self.DEBUG:
            os.makedirs(self.object_output_scan_path, exist_ok=True)
            # Remove all .obj files in the folder
            for file in os.listdir(self.object_output_scan_path):
                if file.endswith(".obj"):
                    os.remove(os.path.join(self.object_output_scan_path, file))
                    self.logger.debug(f"Removed file: {file}")

            for obj in objects:
                obj_vertices = self.mesh_vertices[obj.index_set]

                obj_file_path = os.path.join(self.object_output_scan_path, f"{obj.name.replace(' ', '_')}_{len(obj.index_set)}_vertices.obj")
                with open(obj_file_path, 'w') as obj_file:
                # Write vertices
                    for vertex in obj_vertices:
                        obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                self.logger.debug(f"Saved object {obj.name} to {obj_file_path}")
            
                
    
    def setup_config(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # disable model outputs we don't need, having issues with memory otherwise
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        
        cfg.freeze()
        return cfg
    
    def generate_input_frames(self):
        assert len(self.args.input) == 1, "Only one input directory should be provided"

        if os.path.isdir(self.args.input[0]):
            paths = sorted(glob.glob(os.path.join(self.args.input[0], 'frame_*.jpg')))
            input_frames = [os.path.basename(path).removesuffix('.jpg') for path in paths]
            assert input_frames, f"Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App', the folder only contrains: {os.listdir(self.args.input[0])}"
            
        else:
            raise ValueError("Given input does not exist or is not a directory")
        
        return input_frames, self.args.input[0]

    def project_points_to_image(self, p_in, mvp, image_width, image_height):
        p0 = np.concatenate([p_in, np.ones([p_in.shape[0], 1])], axis=1)
        e0 = np.dot(p0, mvp.T)
        pos_z = e0[:, 2]
        e0 = (e0.T / e0[:, 3]).T
        pos_x = e0[:, 0]
        pos_y = e0[:, 1]
        projections = np.zeros([p_in.shape[0], 3])
        projections[:, 0] = (0.5 + (pos_x) * 0.5) * image_width
        projections[:, 1] = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
        projections[:, 2] = pos_z  # Store the z coordinate
        return projections
    
    
# TODO: 
# Investigate scannet and mean-IoU metric and produce some metric benchmark
# other models mean-IoU: https://paperswithcode.com/sota/semantic-segmentation-on-scannetv2
# add random color palette for objects
# use a BFS to assign lost vertices to nearest object


# DONE:
# use segmentation from mask2former on best perspective image to split image, use current split method on top - check


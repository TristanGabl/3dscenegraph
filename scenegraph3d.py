import argparse
import glob
import multiprocessing as mp
import os

import pickle
import trimesh
import json

import cv2
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config 

from setup_logger import setup_logger


class SceneGraph3D:
    def __init__(
        self,
        args, 
        DEBUG: bool = False,
        SAVE_VISUALIZATION: bool = False,
        FORCE_MASK2FORMER: bool = False,
        SHORTCUT_0: bool = False,
        SHORTCUT_1: bool = False,
    ):
        mp.set_start_method("spawn", force=True)
        self.logger = setup_logger(DEBUG)
        self.DEBUG = DEBUG
        self.logger.info("SceneGraph3D logger initialized")
        self.args = args
        self.logger.info("Arguments: " + str(args))
        self.SAVE_VISUALIZATION = SAVE_VISUALIZATION
        self.logger.info("SAVE_VISUALIZATION: " + str(SAVE_VISUALIZATION))
        self.FORCE_MASK2FORMER = FORCE_MASK2FORMER
        self.logger.info("FORCE_MASK2FORMER: " + str(FORCE_MASK2FORMER))
        self.SHORTCUT_0 = SHORTCUT_0
        self.logger.info("SHORTCUT_0: " + str(SHORTCUT_0))
        self.SHORTCUT_1 = SHORTCUT_1
        self.logger.info("SHORTCUT_1: " + str(SHORTCUT_1))


        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.logger.info("Device: " + str(self.device))
        args.opts += ["MODEL.DEVICE", str(self.device)]
        self.config = self.setup_config(args)

        self.mask2former_predictor = DefaultPredictor(self.config)
        self.metadata = self.mask2former_predictor.metadata
        self.input_image_frames, self.input_scan_path = self.generate_input_frames()
        self.output_scan_path = os.path.join(self.args.output, self.input_scan_path.split('/')[-1])
        self.number_input_image_paths = len(self.input_image_frames)
        self.logger.info("Number image frames: " + str(len(self.input_image_frames)))
    
    def generate_3d_scene_graph(self): # main function
        self.run_mask2former() # first use Mask2Former to get the panoptic segmentations
        # get all classes 
        self.classes = np.unique([segment_info['category_id'] for path in self.output_image_frames for segment_info in pickle.load(open(path + '.pkl', 'rb'))[1]])
        
        # load the mesh vertices
        self.mesh_vertices = np.array(trimesh.load_mesh(os.path.join(self.input_scan_path, 'export_refined.obj')).vertices)
        self.mesh_vertices_votes = np.zeros((self.mesh_vertices.shape[0], len(self.classes)), dtype=int)

        # distribute the panoptic segmentations from the images to the mesh vertices
        self.distribute_panoptic_segmentations() 

        # fuse votes to the vote that is most frequen (global), except for 0 
        self.mesh_vertices_classes = np.apply_along_axis(lambda row: self.classes[np.argmax(row)] if np.any(row) else -1, 1, self.mesh_vertices_votes)

        # plot pointcloud with classes for debugging, TODO: make better visualization
        self.save_segmented_pointcloud() 



    def run_mask2former(self):
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images"
        )

        self.output_image_frames = [] # saved with no suffix
        for frame in self.input_image_frames:
            frame_path = os.path.join(self.input_scan_path, frame)
            image_path = frame_path + '.jpg'
            image_info_path = frame_path + '.json'
            output_image_path = os.path.join(self.output_scan_path, frame)
            inference_output_path = output_image_path + '.pkl'

            image = read_image(image_path, format="RGB")

            if not os.path.exists(inference_output_path) or self.FORCE_MASK2FORMER:
                pbar.set_description("Running Mask2Former on image: {}".format(frame))
                pbar.update()

                image_info = json.load(open(image_info_path, 'r'))
                if image_info['cameraPoseARFrame'] is None or image_info['projectionMatrix'] is None:
                    self.logger.warning("{}: cameraPoseARFrame or projectionMatrix not found in image info json file, skipping this image".format(frame))
                    # remove frame
                    self.input_image_frames.remove(frame)
                    continue

                os.makedirs(self.output_scan_path, exist_ok=True)

                # run inference
                predictions = self.mask2former_predictor(image)

                # save inference output
                with open(inference_output_path, 'wb') as f:
                    pickle.dump((predictions["panoptic_seg"][0].to(torch.device("cpu")), predictions["panoptic_seg"][1]), f)
                
                # save image info
                image_info_relevant = {key: image_info[key] for key in ['cameraPoseARFrame', 'projectionMatrix']}
                with open(output_image_path + '.json', 'w') as f:
                    json.dump(image_info_relevant, f)
        
                if self.SAVE_VISUALIZATION:
                    assert "panoptic_seg" in predictions
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    visualizer = Visualizer(image, 
                                        MetadataCatalog.get(self.config.DATASETS.TEST[0] if len(self.config.DATASETS.TEST) else "__unused"), 
                                        instance_mode=ColorMode.IMAGE
                                        )
                    vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), segments_info)
                    vis_output.save(output_image_path + '.jpg')
                    self.logger.debug("Saved visualization to: {}".format(output_image_path + '.jpg'))
                    
            else:
                pbar.set_description("Skipping image (inference already saved): {}".format(frame))
                pbar.update()

            # if everything is successful, add the path to the list of processed images
            self.output_image_frames.append(output_image_path)
        pbar.close()
            
        self.logger.info("Finished running Mask2former on images\n")

    def distribute_panoptic_segmentations(self):
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images",
        )

        # iterate over the processed images and project the 3d points into the image and apply panoptic segmentation
        for frame in self.output_image_frames:
            pbar.set_description("Projecting points into image and applying panoptic segmentation: {}".format(os.path.basename(frame)))
            
            # load from saved files
            panoptic_seg, segments_info = pickle.load(open(frame + '.pkl', 'rb'))
            image_info = json.load(open(frame + '.json', 'r'))
            image = cv2.imread(frame + '.jpg')

            # compute the model view projection matrix
            pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
            projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
            view_matrix = np.linalg.inv(pose)
            mvp = np.dot(projection_matrix, view_matrix)
            
            # project the 3d point cloud and filter out points that are not in the image
            projections = self.project_points_to_image(self.mesh_vertices, mvp, image.shape[1], image.shape[0])
            projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
            projections[:, :2] = projections_2d # store the z coordinate
            filtered_indices = (projections[:, 0] >= 0) & (projections[:, 0] < image.shape[1]) & (projections[:, 1] >= 0) & (projections[:, 1] < image.shape[0])
            projections_filtered = projections[filtered_indices]

            if self.DEBUG:
                self.logger.debug("number_points_projected: {}".format(len(projections_filtered)))

            # for debugging projected points
            if self.SAVE_VISUALIZATION and not self.SHORTCUT_0:
                img_projected = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # add grey border around the image for debugging
                border_offset = 50
                img_projected = cv2.copyMakeBorder(img_projected, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])
                for i, point in enumerate(projections_filtered[:,:2]):
                    cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 5, (0, 0, 255), -1)

                cv2.imwrite(frame + '_projections.jpg', img_projected) 
                self.logger.debug("saved projections to {}_projections.jpg".format(frame))
            
            # we use panoptic_seg to get votes for object classes for each 3d point
            classes_local_id = np.array([segment_info['id'] for segment_info in segments_info])
            mesh_vertices_classes_local = np.zeros((projections_filtered.shape[0], len(classes_local_id)))
            depth_map = cv2.imread(os.path.join(self.args.input[0],frame.split('/')[-1]).replace("frame", "depth") + ".png", cv2.IMREAD_UNCHANGED)
            depth_map = depth_map / 1000 # convert to meters
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # resize to image size
            # projections_filtered has shape (n, 2) where n is the number of points, depth_map has shape (h, w), i want to get the depth of each point
            depths = depth_map[projections_filtered[:, 1].astype(int), projections_filtered[:, 0].astype(int)]
            # compare depths with projections_filtered[:, 2] to be +- 0.05
            depths_mask = np.zeros(len(projections), dtype=bool)
            depths_mask[filtered_indices] = np.abs(depths - projections_filtered[:, 2]) < 0.05
            for segment_info in segments_info:
                mask = panoptic_seg == segment_info["id"]
                if mask.sum() == 0:
                    continue
                mask = mask[projections_filtered[:, 1], projections_filtered[:, 0]]
                mask = mask & depths_mask[filtered_indices]
                self.mesh_vertices_votes[filtered_indices, np.where(self.classes == segment_info["category_id"])[0][0]] += mask.numpy()
                mesh_vertices_classes_local[:, segment_info["id"]-1] = mask.numpy()

            # fuse votes to the vote that is most frequent except for 0 (local)
            mesh_vertices_classes_local = np.apply_along_axis(lambda row: classes_local_id[np.argmax(row)] if np.any(row) else -1, 1, mesh_vertices_classes_local)
            
            # for debugging point classes
            if self.DEBUG:
                number_of_classes = len(classes_local_id)
                number_of_class_points = np.sum(mesh_vertices_classes_local != 0)
                self.logger.debug("number_of_classes: {}".format(number_of_classes))
                self.logger.debug("number_of_class_points: {}".format(number_of_class_points))
                if self.SAVE_VISUALIZATION and not self.SHORTCUT_1:
                    img_fused_votes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                    # add grey border around the image for debugging
                    border_offset = 50
                    img_fused_votes = cv2.copyMakeBorder(img_fused_votes, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                    # Color points: color each class a different color, no class is black
                    # Define a colormap for the classes
                    colors = np.zeros((len(projections_filtered), 3))
                    
                    if number_of_classes > 0:
                        colormap = plt.cm.get_cmap("tab20", np.max(classes_local_id) + 1)
                    else:
                        colormap = None # no classes found

                    # Map each class to a color
                    for i, class_id in enumerate(classes_local_id):
                        if class_id == -1:
                            colors[mesh_vertices_classes_local == class_id] = (0, 0, 0)
                        else:
                            colors[mesh_vertices_classes_local == class_id] = colormap(class_id)[:3]
                    
                    for i, point in enumerate(projections_filtered[:,:2]):
                        cv2.circle(img_fused_votes, tuple(point.ravel().astype(int) + border_offset), 4, (colors[i] * 255), -1)

                    cv2.imwrite(frame + '_fused_votes.jpg', img_fused_votes)
                    self.logger.debug("saved projections with votes to {}_fused_votes.jpg".format(frame))

            pbar.update()

    
    def project_pointcloud(self, image_info):
        # compute the model view projection matrix
        pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
        projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
        view_matrix = np.linalg.inv(pose)
        mvp = np.dot(projection_matrix, view_matrix)
        
        # project the 3d point cloud and filter out points that are not in the image
        projections = self.project_points_to_image(self.mesh_vertices, mvp, image.shape[1], image.shape[0])
        projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
        projections[:, :2] = projections_2d # store the z coordinate
        filtered_indices = (projections[:, 0] >= 0) & (projections[:, 0] < image.shape[1]) & (projections[:, 1] >= 0) & (projections[:, 1] < image.shape[0])
        projections_filtered = projections[filtered_indices]

    def save_segmented_pointcloud(self):
        np.save(self.args.output + '/' + self.args.input[0].split('/')[-1] + '/pointcloud_classes.npy', self.mesh_vertices_classes)
        self.logger.info("saved pointcloud classes to {}_pointcloud_classes.npy".format(self.args.output))
        
        if self.SAVE_VISUALIZATION:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_vertices[:, 2], c=self.mesh_vertices_classes, cmap='tab20')
            plt.savefig(self.args.output + '/' + self.args.input[0].split('/')[-1] + '/pointcloud_classes.jpg')
            self.logger.debug("saved pointcloud with classes to {}_pointcloud_classes.jpg".format(self.args.output))
        



    
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
            input_image_frames = [os.path.basename(path).removesuffix('.jpg') for path in paths]
            assert input_image_frames, "Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App'"
        else:
            raise ValueError("Given input does not exist or is not a directory")
        
        return input_image_frames, self.args.input[0]

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
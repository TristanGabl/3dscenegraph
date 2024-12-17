import argparse
import glob
import multiprocessing as mp
import os

import pickle
import trimesh
import json

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, '/teamspace/studios/this_studio/Mask2Former')
sys.path.insert(1, '/teamspace/studios/this_studio/Mask2Former/demo')

# change working directory to this folder 
os.chdir('/teamspace/studios/this_studio/3dscenegraph')

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config 
from Mask2Former.demo.predictor import VisualizationDemo


DEBUG_ = True
save_visualization = True
OVERWRITE_0 = True
OVERWRITE_1 = True
OVERWRITE_2 = True

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="3dscenegraph pipeline using mask2former")
    parser.add_argument(
        "--config-file",
        default="../Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file (default set in main.py)",
    )
    
    parser.add_argument(
        "--input",
        nargs=1,
        required=True,
        help="A directory of a scan from '3D Scanner App'; "
        "The directory should be a full export of a scan, containing images and json files etc.",
    )
    parser.add_argument(
        "--output",
        default="output", 
        help="A file or directory to save output visualizations."
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs (default set in main.py); "
        "example: MODEL.WEIGHTS /path/to/model_checkpoint.pkkl",
        default=["MODEL.WEIGHTS", "../Mask2Former/model_weights/model_final_94dc52.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser

def project_point_mvp(p_in, mvp, image_width, image_height):
    p0 = np.concatenate([p_in, np.ones([p_in.shape[0], 1])], axis=1)
    e0 = np.dot(p0, mvp.T)
    e0 = (e0.T / e0[:, 3]).T
    pos_x = e0[:, 0]
    pos_y = e0[:, 1]
    projections = np.zeros([p_in.shape[0], 2])
    projections[:,0] = (0.5 + (pos_x) * 0.5) * image_width
    projections[:,1] = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
    return projections


def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if torch.cuda.is_available():
        args.opts = ["MODEL.DEVICE", "cuda"] + args.opts
        device = "cuda"
    else:
        args.opts = ["MODEL.DEVICE", "cpu"] + args.opts
        device = "cpu"

    cfg = setup_cfg(args)

    mask2former_predictor = DefaultPredictor(cfg)
    mask2former_predictor.model.to(device)
    mask2former_predictor.model.eval()


    if os.path.isdir(args.input[0]):
        input_images = glob.glob(os.path.expanduser(args.input[0] + '/frame_*.jpg'))
        assert input_images, "Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App'"
    
    input_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print("\n\n")
    processed_image_paths = [] # saved with no suffix
    for path in tqdm.tqdm(input_images):
        print()
        image = read_image(path, format="BGR") # use BGR format for OpenCV compatibility (not really used here)
        start_time = time.time()

        output_image_path = os.path.join(args.output, *path.split('/')[-2:]).removesuffix('.jpg')
        path_inference_output = output_image_path + '.pkl'
        image_info_path = path.removesuffix('.jpg') + '.json'

        if not os.path.exists(path_inference_output) or OVERWRITE_0:
            os.makedirs(os.path.dirname(path_inference_output), exist_ok=True)

            # run inference
            with torch.no_grad():
                predictions = mask2former_predictor(image)
        

            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]
        
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} panoptic segments".format(len(predictions["panoptic_seg"]))
                    if "panoptic_seg" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            # save inference output
            with open(output_image_path + '.pkl', 'wb') as f:
                pickle.dump((predictions["panoptic_seg"][0].to(torch.device("cpu")), predictions["panoptic_seg"][1]), f)
            
            # save image info
            image_data = json.load(open(image_info_path, 'r'))
            if image_data['cameraPoseARFrame'] is None or image_data['projectionMatrix'] is None:
                logger.error("{}: cameraPoseARFrame or projectionMatrix not found in image info json file".format(path))
                continue
            image_data_relevant = {key: image_data[key] for key in ['cameraPoseARFrame', 'projectionMatrix']}
            with open(output_image_path + '.json', 'w') as f:
                json.dump(image_data_relevant, f)
    
            if save_visualization and not os.path.exists(output_image_path + '.jpg'):
                assert "panoptic_seg" in predictions
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                visualizer = Visualizer(image, 
                                    MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"), 
                                    instance_mode=ColorMode.IMAGE
                                    )
                vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), segments_info)
                vis_output.save(output_image_path + '.jpg')
            
            
        else:
            logger.info("{}: found pre-processed, skipping".format(path))

        # if everything is successful, add the path to the list of processed images
        processed_image_paths.append(output_image_path)
        
    logger.info("Finished panoptic segmentation inference\n") # -----------------------------------------------

    # get all classes in advance
    classes = np.unique([segment_info['category_id'] for path in processed_image_paths for segment_info in pickle.load(open(path + '.pkl', 'rb'))[1]])

    # load the mesh vertices
    mesh_vertices = np.array(trimesh.load_mesh(os.path.join(args.input[0], 'export_refined.obj')).vertices)
    mesh_vertices_votes = np.zeros((mesh_vertices.shape[0], len(classes)), dtype=int)
    for path in tqdm.tqdm(processed_image_paths):
        print()
        # load from saved files
        panoptic_seg, segments_info = pickle.load(open(path + '.pkl', 'rb'))
        image_data = json.load(open(path + '.json', 'r'))
        image = cv2.imread(path + '.jpg')

        # compute the model view projection matrix
        pose = np.array(image_data['cameraPoseARFrame']).reshape((4, 4))
        projection_matrix = np.array(image_data['projectionMatrix']).reshape((4, 4))
        view_matrix = np.linalg.inv(pose)
        mvp = np.dot(projection_matrix, view_matrix)
        
        # project the 3d point cloud and filter out points that are not in the image
        projections = project_point_mvp(mesh_vertices, mvp, image.shape[1], image.shape[0])
        projections = np.round(projections).astype(int) # round to nearest pixel
        filtered_indices = (projections[:, 0] >= 0) & (projections[:, 0] < image.shape[1]) & (projections[:, 1] >= 0) & (projections[:, 1] < image.shape[0])
        projections_filtered = projections[filtered_indices]

        # for debugging projected points
        if DEBUG_:
            number_points_projected_filtered = len(projections_filtered)
            logger.info("number_points_projected: {}".format(number_points_projected_filtered))
            if save_visualization and not os.path.exists(path + '_projections.jpg') or OVERWRITE_1:
                img_projected = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # add grey border around the image for debugging
                border_offset = 50
                img_projected = cv2.copyMakeBorder(img_projected, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])
                for i, point in enumerate(projections_filtered):
                    cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 5, (0, 0, 255), -1)

                cv2.imwrite(path + '_projections.jpg', img_projected) 
                logger.info("saved projections to {}_projections.jpg".format(path))   
        
        else:
            logger.info(
                "{}: found pre-processed, skipping".format(
                    path
                )
            )
        
        # we use panoptic_seg to get votes for object classes for each 3d point
        classes_local_id = np.array([segment_info['id'] for segment_info in segments_info])
        mesh_vertices_classes_local = np.zeros((projections_filtered.shape[0], len(classes_local_id)))
        for segment_info in segments_info:
            mask = panoptic_seg == segment_info["id"]
            if mask.sum() == 0:
                continue
            mask = mask[projections_filtered[:, 1], projections_filtered[:, 0]]
            mesh_vertices_votes[filtered_indices, np.where(classes == segment_info["category_id"])[0][0]] += mask.numpy()
            mesh_vertices_classes_local[:, segment_info["id"]-1] = mask.numpy()

         
        # fuse votes to the vote that is most frequent except for 0 (local)
        mesh_vertices_classes_local = np.apply_along_axis(lambda row: classes_local_id[np.argmax(row)] if np.any(row) else -1, 1, mesh_vertices_classes_local)
        
        # for debugging point classes
        if DEBUG_:
            number_of_classes = len(classes_local_id)
            number_of_class_points = np.sum(mesh_vertices_classes_local != 0)
            logger.info("number_of_classes: {}".format(number_of_classes))
            logger.info("number_of_class_points: {}".format(number_of_class_points))
            if save_visualization and not os.path.exists(path + '_fused_votes.jpg') or OVERWRITE_2:
                img_fused_votes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
                # add grey border around the image for debugging
                border_offset = 50
                img_fused_votes = cv2.copyMakeBorder(img_fused_votes, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                # Color points: color each class a different color, no class is black
                # Define a colormap for the classes
                colors = np.zeros((len(projections_filtered), 3))
                colormap = plt.cm.get_cmap("tab10", np.max(classes_local_id) + 1)

                # Map each class to a color
                for i, class_id in enumerate(classes_local_id):
                    if class_id == -1:
                        colors[mesh_vertices_classes_local == class_id] = (0, 0, 0)
                    else:
                        colors[mesh_vertices_classes_local == class_id] = colormap(class_id)[:3]
                
                for i, point in enumerate(projections_filtered):
                    cv2.circle(img_fused_votes, tuple(point.ravel().astype(int) + border_offset), 4, (colors[i] * 255), -1)

                cv2.imwrite(path + '_fused_votes.jpg', img_fused_votes)
                logger.info("saved fused votes to {}_fused_votes.jpg".format(path))

    
    # fuse votes to the vote that is most frequent except for 0 (global)
    mesh_vertices_classes = np.apply_along_axis(lambda row: classes[np.argmax(row)] if np.any(row) else -1, 1, mesh_vertices_votes)

    # plot pointcloud with classes for debugging
    if DEBUG_:
        if (save_visualization and not os.path.exists(args.output + '/pointcloud_classes.jpg')) or OVERWRITE_1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], c=mesh_vertices_classes, cmap='tab20')
            plt.savefig(args.output + '/' + args.input[0].split('/')[-1] + '/pointcloud_classes.jpg')
            logger.info("saved pointcloud with classes to {}_pointcloud_classes.jpg".format(args.output))
            # save pointcloud classes to a file
            np.save(args.output + '/' + args.input[0].split('/')[-1] + '/pointcloud_classes.npy', mesh_vertices_classes)
    
            
# try to run demo on daniel_example_scan to debug shutdown error




if __name__ == "__main__":
    main()
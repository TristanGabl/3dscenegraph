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

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config 
from Mask2Former.demo.predictor import VisualizationDemo
import torch

DEBUG_ = True
save_visualization = True

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
    else:
        args.opts = ["MODEL.DEVICE", "cpu"] + args.opts

    cfg = setup_cfg(args)

    mask2former_predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.input[0]):
        input_images = glob.glob(os.path.expanduser(args.input[0] + '/frame_*.jpg'))
        assert input_images, "Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App'"
    

    print("\n\n")
    processed_image_paths = [] # saved with no suffix
    for path in tqdm.tqdm(input_images, disable=not args.output):
        image = read_image(path, format="BGR") # use BGR format for OpenCV compatibility (not really used here)
        start_time = time.time()

        output_image_path = os.path.join(args.output, *path.split('/')[-2:]).removesuffix('.jpg')
        path_inference_output = output_image_path + '.pkl'
        image_info_path = path.removesuffix('.jpg') + '.json'

        if not os.path.exists(path_inference_output):
            os.makedirs(os.path.dirname(path_inference_output), exist_ok=True)

            # run inference
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
            with open(path_inference_output, 'wb') as f:
                pickle.dump(predictions["panoptic_seg"], f)
            
            # save image info
            image_data = json.load(open(image_info_path, 'r'))
            if image_data['cameraPoseARFrame'] is None or image_data['projectionMatrix'] is None:
                logger.error(
                    "{}: cameraPoseARFrame or projectionMatrix not found in image info json file".format(
                        path
                    )
                )
                continue
            image_data_relevant = {key: image_data[key] for key in ['cameraPoseARFrame', 'projectionMatrix']}
            with open(output_image_path + '.json', 'w') as f:
                json.dump(image_data_relevant, f, indent=1) 
    
            if save_visualization and not os.path.exists(output_image_path + '.jpg'):
                assert "panoptic_seg" in predictions
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                visualizer = Visualizer(image, 
                                    MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"), 
                                    instance_mode=ColorMode.IMAGE
                                    )
                vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), segments_info)
                vis_output.save(output_image_path + '.jpg')
            
            # load json file
            image_info = json.load(open(image_info_path, 'r'))


        else:
            logger.info(
                "{}: found pre-processed, skipping".format(
                    path
                )
            )

        # if everything is successful, add the path to the list of processed images
        processed_image_paths.append(output_image_path)
        
        # predictions.keys() = dict_keys(['sem_seg', 'panoptic_seg', 'instances'])
        # type(predictions("instances")) = detectron2.structures.instances.Instances
        # debug predictions["panoptic_seg"][0]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["panoptic_seg"][0].cpu().numpy().astype(np.uint8))
        # debug predictions["sem_seg"]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["sem_seg"][0].cpu().numpy().astype(np.uint8))
        
    # project 3d point cloud into each 2d image and record the panoptic segmentation onto the 3d point cloud

    # load the 3d point cloud
    mesh_vertices = np.array(trimesh.load(os.path.join(args.input[0], 'export_refined.obj')).vertices)

    for path in processed_image_paths:
        # load the 2d image
        panoptic_seg, segments_info = pickle.load(open(path + '.pkl', 'rb'))
        image_data = json.load(open(path + '.json', 'r'))
        image = cv2.imread(path + '.jpg')
    
        if DEBUG_:
            print("image.shape = ", image.shape)
            print("panoptic_seg.shape = ", panoptic_seg.shape)
            print("len(np.unique(panoptic_seg))-1 = ", len(np.unique(panoptic_seg))-1)
            print("len(segments_info) = ", len(segments_info))
            print("segments_info: ", segments_info)


        # Load the 3d point cloud
        # Load the panoptic segmentation
        # Project the panoptic segmentation onto the 3d point cloud
        # Save the 3d point cloud with the panoptic segmentation
        pose = np.array(image_data['cameraPoseARFrame']).reshape((4, 4))
        projection_matrix = np.array(image_data['projectionMatrix']).reshape((4, 4))
        view_matrix = np.linalg.inv(pose)
        mvp = np.dot(projection_matrix, view_matrix)
        
        # project the 3d point cloud
        projections = project_point_mvp(mesh_vertices, mvp, image.shape[1], image.shape[0])
        # projections = projections.astype(int)
        # filter out points that are outside the image
        #projections = projections[(projections[:,0] >= 0) & (projections[:,0] < image.shape[1]) & (projections[:,1] >= 0) & (projections[:,1] < image.shape[0])]

        if DEBUG_:
            print("projections.shape = ", projections.shape)
            print("projections = ", projections)
            # visualize the projections by drawing circles on the image
            # Draw the projected points
            img_projected = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            point_number1 = 0
            for point in projections:
                if point[0] < 0 or point[1] < 0 or point[0] >= image.shape[1] or point[1] >= image.shape[0]:
                    continue
                point_number1 += 1
                cv2.circle(img_projected, tuple(point.ravel().astype(int)), 5, (255, 0, 0), -1)

            cv2.imwrite(path + '_projections.jpg', img_projected)
            print("point_number1 = ", point_number1)
            print("saved projections to ", path + '_projections.jpg')




        



if __name__ == "__main__":
    main()
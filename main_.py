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

from scenegraph3d import SceneGraph3D


DEBUG_ = True
save_visualization = True
FORCE_MASK2FORMER = False # if True, the mask2former model will be run even if the processed images already exist
OVERWRITE_1 = True
OVERWRITE_2 = True


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
        default=["MODEL.WEIGHTS", "/teamspace/studios/this_studio/Mask2Former/model_weights/model_final_94dc52.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()

    pipeline = SceneGraph3D(args, DEBUG_, save_visualization, FORCE_MASK2FORMER, OVERWRITE_1, OVERWRITE_2)

    pipeline.generate_3d_scene_graph()
    
    
    # fuse votes to the vote that is most frequent except for 0 (global)
    mesh_vertices_classes = np.apply_along_axis(lambda row: classes[np.argmax(row)] if np.any(row) else -1, 1, mesh_vertices_votes)

    # plot pointcloud with classes for debugging, TODO: make better visualization
    if DEBUG_:
        if (save_visualization and not os.path.exists(args.output + '/pointcloud_classes.jpg')) or OVERWRITE_1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], c=mesh_vertices_classes, cmap='tab20')
            plt.savefig(args.output + '/' + args.input[0].split('/')[-1] + '/pointcloud_classes.jpg')
            logger.debug("saved pointcloud with classes to {}_pointcloud_classes.jpg".format(args.output))
            # save pointcloud classes to a file
            np.save(args.output + '/' + args.input[0].split('/')[-1] + '/pointcloud_classes.npy', mesh_vertices_classes)




if __name__ == "__main__":
    main()
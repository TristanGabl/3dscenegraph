import argparse
import glob
import multiprocessing as mp
import os

import pickle
import trimesh
import json

import sys
sys.path.insert(1, os.path.join(sys.path[0], './Mask2Former'))
# change working directory to this folder 
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


DEBUG = False
save_visualization = True
save_objects = False
FORCE_MASK2FORMER = False # if True, the mask2former model will be run even if the processed images already exist
SHORTCUT_0 = True # if True, generating frame_XXXXX_projections.jpg will be skipped
SHORTCUT_1 = True # if True, generating frame_XXXXX_fused_votes.jpg will be skipped

USE_LLM = False



def get_parser():
    parser = argparse.ArgumentParser(description="3dscenegraph pipeline using mask2former")
    parser.add_argument(
        "--config-file",
        default="helper_repos/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
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
        default=["MODEL.WEIGHTS", "model_weights/model_final_f07440.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()

    pipeline = SceneGraph3D(args, DEBUG, save_visualization, save_objects, FORCE_MASK2FORMER, SHORTCUT_0, SHORTCUT_1, USE_LLM)

    pipeline.generate_3d_scene_graph()

    # TODO: remember images of objects to use create mask of object and double check non-duplicate property by running mask2former on the objectm ask again
 

if __name__ == "__main__":
    main()
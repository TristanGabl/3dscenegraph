import argparse
import glob
import multiprocessing as mp
import os

import pickle
import trimesh

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

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    save_visualization = True


    if torch.cuda.is_available():
        args.opts = ["MODEL.DEVICE", "cuda"] + args.opts
    else:
        args.opts = ["MODEL.DEVICE", "cpu"] + args.opts

    cfg = setup_cfg(args)

    mask2former_predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.input[0]):
        input_images = glob.glob(os.path.expanduser(args.input[0] + '/frame_*.jpg'))
        assert input_images, "Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App'"
    
    # we extract the numbers from the input image paths (exmaple: frame_00188.jpg) and create a copy of args.input with the paths from the json file

    print("\n\n")
    processed_input_images = []
    for path in tqdm.tqdm(input_images, disable=not args.output):
        # use PIL, to be consistent with evaluation
        image = read_image(path, format="BGR")
        start_time = time.time()

        path_image_output = os.path.join(args.output, *path.split('/')[-2:])
        processed_input_images.append(path_image_output)
        path_inference_output = path_image_output.replace('.jpg', '.pkl')

        if not os.path.exists(path_inference_output):
            # create folder if not exists
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
            
            if save_visualization:
                assert "panoptic_seg" in predictions
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                visualizer = Visualizer(image, 
                                    MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"), 
                                    instance_mode=ColorMode.IMAGE
                                    )
                vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), segments_info)
                vis_output.save(os.path.join(args.output, *path.split('/')[-2:]))
        else:
            logger.info(
                "{}: found pre-processed, skipping".format(
                    path
                )
            )
        
        # predictions.keys() = dict_keys(['sem_seg', 'panoptic_seg', 'instances'])
        # type(predictions("instances")) = detectron2.structures.instances.Instances
        # debug predictions["panoptic_seg"][0]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["panoptic_seg"][0].cpu().numpy().astype(np.uint8))
        # debug predictions["sem_seg"]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["sem_seg"][0].cpu().numpy().astype(np.uint8))
        
    # project 3d point cloud into each 2d image and record the panoptic segmentation onto the 3d point cloud

    # load the 3d point cloud
    mesh = trimesh.load(os.path.join(args.input[0], 'export_refined.obj'))


    for processed_input_image in processed_input_images:
        print(processed_input_image)
        # Load the 3d point cloud
        # Load the panoptic segmentation
        # Project the panoptic segmentation onto the 3d point cloud
        # Save the 3d point cloud with the panoptic segmentation
        



if __name__ == "__main__":
    main()
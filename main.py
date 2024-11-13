import argparse
import glob
import multiprocessing as mp
import os

import pickle

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
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    
    parser.add_argument(
        "--input_images",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output", 
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
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

    create_pre_processed = True

    if torch.cuda.is_available():
        args.opts = ["MODEL.DEVICE", "cuda"] + args.opts
    else:
        args.opts = ["MODEL.DEVICE", "cpu"] + args.opts

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if len(args.input_images) == 1:
        args.input_images = glob.glob(os.path.expanduser(args.input_images[0]))
        assert args.input_images, "The input path(s) was not found"
    
    # we extract the numbers from the input image paths (exmaple: frame_00188.jpg) and create a copy of args.input with the paths from the json file

    # create list to store the json paths
    json_paths = []
    for path in args.input_images: 
        json_path = path.replace('jpg', 'json')
        json_paths.append(json_path)

    panoptic_segs = []
    for path in tqdm.tqdm(args.input_images, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)

        # predictions.keys() = dict_keys(['sem_seg', 'panoptic_seg', 'instances'])
        # type(predictions("instances")) = detectron2.structures.instances.Instances
        # debug predictions["panoptic_seg"][0]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["panoptic_seg"][0].cpu().numpy().astype(np.uint8))
        # debug predictions["sem_seg"]: cv2.imwrite(os.path.join(os.getcwd(), 'output', 'debug{}.png'.format(path.split('/')[-1].replace('.jpg', ''))), 100 * predictions["sem_seg"][0].cpu().numpy().astype(np.uint8))

        # save panoptic segmentation
        panoptic_seg = predictions["panoptic_seg"][0].cpu().numpy()
        if (create_pre_processed):
            with open(args.output.split("/")[:-1] + "/small_3DScannerApp_export" + path.split('/')[-1].replace('.jpg', '.pkl'), 'wb') as f:
                pickle.dump(panoptic_seg, f)
            

        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, "segmented_" + os.path.basename(path))
            else:
                assert len(args.input_images) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)

if __name__ == "__main__":
    main()
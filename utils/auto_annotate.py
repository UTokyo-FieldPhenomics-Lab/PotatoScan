# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2023-09-06 14:22:36

## Use a trained PointRend model to auto-annotate unlabelled images

## general libraries
import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")

## detectron2-libraries 
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

## import PointRend project
from detectron2.projects import point_rend

## libraries for preparing the datasets
from mrcnn_tools import list_files, visualize_mrcnn, write_darwin_annotations, write_labelme_annotations


if __name__ == "__main__":
    ## Initialize the args_parser and some variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='datasets/train', help='the folder with images that need to be auto-annotated')
    parser.add_argument('--img_separator', type=str, default='rgb', help='string-identifier to separate the relevant frames from the non-relevant frames')
    parser.add_argument('--network_config', type=str, default='./detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml', help='specify the backbone of the CNN')
    parser.add_argument('--classes', nargs="+", default=[], help='a list with all the classes')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold for the Mask R-CNN inference')
    parser.add_argument('--nms_thres', type=float, default=0.2, help='non-maximum suppression threshold for the Mask R-CNN inference')
    parser.add_argument('--weights_file', type=str, default=[], help='weight-file (.pth)')
    parser.add_argument('--export_format', type=str, default='labelme', help='Choose either "labelme", "darwin"')

    ## Load the args_parser and initialize some variables
    opt = parser.parse_args()
    print(opt)
    print()

    use_coco = False
    images, annotations = list_files(opt.img_dir, opt.img_separator)

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(opt.network_config)
    cfg.NUM_GPUS = 1

    if opt.weights_file.lower().endswith(".yaml"):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(opt.weights_file)
        use_coco = True
    elif opt.weights_file.lower().endswith((".pth", ".pkl")):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(opt.classes)
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(opt.classes)
        cfg.MODEL.WEIGHTS = opt.weights_file

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf_thres
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = opt.nms_thres
    cfg.DATALOADER.NUM_WORKERS = 0

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    predictor = DefaultPredictor(cfg)

    for i in tqdm(range(len(images))):
        imgname = images[i]
        basename = os.path.basename(imgname)
        img = cv2.imread(os.path.join(opt.img_dir, imgname))
        height, width, _ = img.shape

        # Do the image inference and extract the outputs from Mask R-CNN
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()

        if use_coco:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            class_labels = v.metadata.get("thing_classes", None)
        else:
            class_labels = opt.classes

        class_names = []
        for h in range(len(classes)):
            class_id = classes[h]
            class_name = class_labels[class_id]
            class_names.append(class_name)

        img_vis = visualize_mrcnn(img, classes, scores, masks, boxes, class_labels)

        if opt.export_format == "labelme":
            write_labelme_annotations(opt.img_dir, basename, class_names, masks, height, width)

        if opt.export_format == "darwin":
            write_darwin_annotations(opt.img_dir, basename, class_names, masks, height, width) 
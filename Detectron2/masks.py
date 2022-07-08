path = '/home/yln1kor/Downloads/kitti_official_semantic/training'
path_images = path + '/image_2'
path_instance = path + '/instance'
path_semantic = path + '/semantic_rgb'
import glob
import os
import cv2
import numpy as np
import warnings as wr
wr.filterwarnings("ignore")

gt_masks = []
c = 1

for imageName in sorted(glob.glob(os.path.join(path_semantic, '*.png'))):
    im = cv2.imread(imageName)
    mask = (im == [142,0,0]).all(-1)
    gt_masks.append(mask)
    c += 1
    if c == 10:
        break

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

c =1
it = 0
im_predmasks = []
for imageName in sorted(glob.glob(os.path.join(path_images, '*.png'))):
    im = cv2.imread(imageName)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = outputs["instances"][outputs["instances"].pred_classes == 2].pred_masks.to('cpu')
    out_img = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 2].to("cpu"))
    out = out.numpy()
    c += 1
    pred_mask = np.full(out[0].shape,False, dtype =bool)

    for j in out:
        pred_mask = np.logical_or(pred_mask,j)
    im_predmasks.append(pred_mask)
    cv2.imshow('pred_mask', pred_mask.astype(np.uint8)*255)
    cv2.imshow('gt',gt_masks[it].astype(np.uint8)*255)
    cv2.waitKey(0)
    it += 1
    if c == 10:
        break

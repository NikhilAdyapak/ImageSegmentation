path = '/home/yln1kor/Downloads/kitti_official_semantic/training'
path_images = path + '/image_2'
path_instance = path + '/instance'
path_semantic = path + '/semantic_rgb'
import glob
import os
import cv2
import numpy as np
import warnings as wr

from torch import logical_and
wr.filterwarnings("ignore")

gt_masks = []
c = 1
for imageName in sorted(glob.glob(os.path.join(path_semantic, '*.png'))):
    im = cv2.imread(imageName)
    mask = (im == [142,0,0]).all(-1)
    gt_masks.append(mask)
    # print(mask)
    c += 1
    # cv2.imshow('img',im)
    # cv2.imshow('mask',mask.astype(np.uint8)*255)
    # cv2.waitKey(0)
    if c == 10:
        break

# !python -m pip install pyyaml==5.1
# # Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# # so we install from source instead. This takes a few minutes.
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Install pre-built detectron2 that matches pytorch version, if released:
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# #!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html
# python3 -m pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html

# # exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime

# print(gt_masks)
# print('\n\n')
# print(len(gt_masks))

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches 
# import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

c =1
it = 0
im_predmasks = []
for imageName in sorted(glob.glob(os.path.join(path_images, '*.png'))):
    im = cv2.imread(imageName)
    outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = outputs["instances"][outputs["instances"].pred_classes == 2].pred_masks.to('cpu')
    out_img = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 2].to("cpu"))
    out = out.numpy()
    c += 1
    # cv2.imshow('img',im)
    # cv2.imshow('pred',out_img.get_image())
    pred_mask = np.full(out[0].shape,False, dtype =bool)
    # print(pred_mask)
    # print(out[0].shape)
    for j in out:
    #     cv2.imshow('mask',j.astype(np.uint8)*255)
    #     cv2.waitKey(0)
        pred_mask = np.logical_or(pred_mask,j)
    im_predmasks.append(pred_mask)
    # cv2.imshow('pred_mask', pred_mask.astype(np.uint8)*255)
    # cv2.imshow('gt',gt_masks[it].astype(np.uint8)*255)
    # cv2.waitKey(0)
    it += 1
    if c == 10:
        break


for i in range(len(gt_masks)):
    gt = gt_masks[i]
    pred = im_predmasks[i]
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.sum(intersection) / np.sum(union)
    Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    print('IOU:',IOU, 'Dice Coeff:',Dice_coeff)

# print(im_predmasks)
# print('\n\n')
# print(len(im_predmasks))

# #print(gt_masks[0]*1)
# #print(type(gt_masks[0]*1))


# import tensorflow as tf
# from keras import backend as K
# def iou_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#     union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#     iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#     return iou

# for i in range(len(gt_masks)):
#     gt = tf.convert_to_tensor(gt_masks[i] * 1)
#     pred = tf.convert_to_tensor(im_predmasks[i] * 1)
#     print(iou_coef(gt,pred, smooth = 1))


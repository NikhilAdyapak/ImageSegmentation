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

gt_masks_cars = []
gt_masks_persons = []
c = 1
def Metrics(gt,pred):
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.sum(intersection) / np.sum(union)
    Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    return((IOU,Dice_coeff))


for imageName in sorted(glob.glob(os.path.join(path_semantic, '*.png'))):
    im = cv2.imread(imageName)
    mask_cars = (im == [142,0,0]).all(-1)
    mask_persons = (im == [60,20,220]).all(-1)
    gt_masks_cars.append(mask_cars)
    gt_masks_persons.append(mask_persons)
    # print(mask)
    c += 1
    # cv2.imshow('img',im)
    # cv2.imshow('mask',mask.astype(np.uint8)*255)
    # cv2.waitKey(0)
    if c == 10:
        break

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
im_predmasks_cars = []
im_predmasks_persons = []
for imageName in sorted(glob.glob(os.path.join(path_images, '*.png'))):
    im = cv2.imread(imageName)
    outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out_cars = outputs["instances"][outputs["instances"].pred_classes == 2].pred_masks.to('cpu')
    out_persons = outputs["instances"][outputs["instances"].pred_classes == 0].pred_masks.to('cpu')
    out_img = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 2].to("cpu"))
    out_cars = out_cars.numpy()
    out_persons = out_persons.numpy()
    c += 1
    # cv2.imshow('img',im)
    # cv2.imshow('pred',out_img.get_image())
    pred_mask_cars = np.full(out_cars[0].shape,False, dtype = bool)
    pred_mask_persons = np.full(out_cars[0].shape,False, dtype = bool)
    # print(pred_mask)
    # print(out[0].shape)
    for j in out_cars:
        # cv2.imshow('mask',j.astype(np.uint8)*255)
        # cv2.waitKey(0)
        pred_mask_cars = np.logical_or(pred_mask_cars,j)
    im_predmasks_cars.append(pred_mask_cars)

    for j in out_persons:
        # cv2.imshow('mask',j.astype(np.uint8)*255)
        # cv2.waitKey(0)
        pred_mask_persons = np.logical_or(pred_mask_persons,j)
    im_predmasks_persons.append(pred_mask_persons)
    cv2.imshow('pred_mask', pred_mask_cars.astype(np.uint8)*255)
    cv2.imshow('gt',gt_masks_cars[it].astype(np.uint8)*255)
    cv2.waitKey(0)
    cv2.imshow('pred_mask', pred_mask_persons.astype(np.uint8)*255)
    cv2.imshow('gt',gt_masks_persons[it].astype(np.uint8)*255)
    cv2.waitKey(0)
    it += 1
    if c == 10:
        break


car = []
person = []
for i in range(len(gt_masks_persons)):
    gt_cars = gt_masks_cars[i]
    gt_persons = gt_masks_persons[i]
    pred_cars = im_predmasks_cars[i]
    pred_persons = im_predmasks_persons[i]
    car.append(Metrics(gt_cars,pred_cars))
    person.append(Metrics(gt_persons,pred_persons))

print('Cars\n','(IOU,DSC)')
for i in car:
    print(i)
print('\n')
print('Persons\n','(IOU,DSC)')
for i in person:
    print(i)



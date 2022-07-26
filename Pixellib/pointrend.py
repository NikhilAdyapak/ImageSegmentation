import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import numpy as np
import warnings as wr
import cv2
import glob
import os

wr.filterwarnings("ignore")

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")

path = '/home/yln1kor/Downloads/kitti_official_semantic/training'
path_images = path + '/image_2'
path_instance = path + '/instance'
path_semantic = path + '/semantic_rgb'

gt_masks = []
c = 1
for imageName in sorted(glob.glob(os.path.join(path_semantic, '*.png'))):
    im = cv2.imread(imageName)
    mask = (im == [142,0,0]).all(-1)
    gt_masks.append(mask)
    c += 1
    if c == 10:
        break

import numpy as np
import os, cv2


c =1
it = 0
im_predmasks = []
for imageName in sorted(glob.glob(os.path.join(path_images, '*.png'))):
    results, output = ins.segmentImage(imageName, show_bboxes=True)
    masks = results["masks"]
    masks = masks.transpose((2, 0, 1))
    c += 1
    pred_mask = np.full(masks[0].shape,False, dtype =bool)
    for j in range(len(masks)):
        if results["class_names"][j] == 'car':
            pred_mask = np.logical_or(pred_mask,masks[j])
    im_predmasks.append(pred_mask)
    cv2.imshow('pred_mask', pred_mask.astype(np.uint8)*255)
    cv2.imshow('gt',gt_masks[it].astype(np.uint8)*255)
    cv2.waitKey(0)
    it += 1
    if c == 10:
        break

sum_IOU = 0
sum_DSC = 0
for i in range(len(gt_masks)):
    gt = gt_masks[i]
    pred = im_predmasks[i]
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.sum(intersection) / np.sum(union)
    sum_IOU += IOU
    Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    sum_DSC += Dice_coeff
    # print('IOU:',IOU, 'Dice Coeff:',Dice_coeff)

print('IOU',sum_IOU/len(gt_masks))
print('DSC',sum_DSC/len(gt_masks))

# with open('ptrend_masks.npy', 'wb') as f:
#     np.save(f,im_predmasks)
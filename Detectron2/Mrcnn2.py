# !python -m pip install pyyaml==5.1
# # Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# # so we install from source instead. This takes a few minutes.
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Install pre-built detectron2 that matches pytorch version, if released:
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# #!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html
# python3 -m pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html

# # exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime

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


for i in range(1,10):
    img = '/home/yln1kor/nikhil-test/archive/Test/Test/JPEGImages/image (' + str(i) + ').jpg'
    im = cv2.imread(img)
    outputs = predictor(im)

    print('\n\n')
    print(i)
    pred = [0,2]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out_boxes = []
    for j in pred:
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == j].to("cpu"))
        boxes = v._convert_boxes(outputs["instances"][outputs['instances'].pred_classes == j].pred_boxes.to('cpu'))
        for box in boxes:
            box = [round(box[0]), round(box[1]), round(box[2]) , round(box[3])]
            print(box)
    cv2.imshow("output",out.get_image())
    cv2.waitKey(0)
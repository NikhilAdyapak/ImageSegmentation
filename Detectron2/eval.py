def Maskgen_model(outputs, ind):
    out = outputs["instances"][outputs["instances"].pred_classes == ind].pred_masks.to('cpu')
    out = out.numpy()

    pred_mask = np.full(out[0].shape,False, dtype =bool)
    for j in out:
        pred_mask = np.logical_or(pred_mask,j)
    return(pred_mask)

def Maskgen(image, color_code):
    mask = (image == color_code).all(-1)
    return(mask)


def Metrics(gt,pred):
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.sum(intersection) / np.sum(union)
    Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    return((IOU,Dice_coeff))


class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

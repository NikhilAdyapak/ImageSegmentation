import numpy as np

def ensemble(gt_mask, model1_mask, model2_mask):
    pred_mask = np.full(gt_mask.shape,False, dtype =bool)
    pred_mask1 = np.logical_or(pred_mask,model1_mask)
    pred_mask1 = np.logical_or(pred_mask,model2_mask)
    pred_mask2 = np.logical_and(pred_mask,model2_mask)

    gt = gt_mask
    pred = pred_mask1
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU1 = np.sum(intersection) / np.sum(union)
    Dice_coeff1 = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    
    pred = pred_mask2
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU2 = np.sum(intersection) / np.sum(union)
    Dice_coeff2 = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    
    # 2 Ensemble techniques, (IOU1, Dice_coeff1) & (IOU2, Dice_coeff2)
    # IOU1, Dice_coeff1 < IOU2, Dice_coeff2
    return([(IOU1,Dice_coeff1),(IOU2,Dice_coeff2)])
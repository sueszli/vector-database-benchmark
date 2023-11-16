import numpy as np

def Cal_IoU(GT_bbox, Pred_bbox):
    if False:
        return 10
    '\n    Args:\n        GT_bbox:  the bounding box of the ground truth\n        Pred_bbox: the bounding box of the predicted\n    Returns:\n        IoU: Intersection over Union\n    '
    ixmin = max(GT_bbox[0], Pred_bbox[0])
    iymin = max(GT_bbox[1], Pred_bbox[1])
    ixmax = min(GT_bbox[2], Pred_bbox[2])
    iymax = min(GT_bbox[3], Pred_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    area = iw * ih
    S1 = (Pred_bbox[2] - GT_bbox[0] + 1) * (Pred_bbox[3] - GT_bbox[1] + 1)
    S2 = (GT_bbox[2] - GT_bbox[0] + 1) * (GT_bbox[3] - GT_bbox[1] + 1)
    S = S1 + S2 - area
    iou = area / S
    return iou
if __name__ == '__main__':
    pred_bbox = np.array([40, 40, 100, 100])
    gt_bbox = np.array([70, 80, 110, 130])
    print(Cal_IoU(pred_bbox, gt_bbox))
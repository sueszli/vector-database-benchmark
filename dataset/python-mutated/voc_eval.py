"""
The mAP evaluation script and various util functions are adapted from:
https://github.com/rbgirshick/py-faster-rcnn/commit/45e0da9a246fab5fd86e8c96dc351be7f145499f
"""
from __future__ import print_function
import numpy as np
GT_CLASS_INDEX = 4
GT_DIFFICULT_INDEX = 5
GT_DETECTED_INDEX = 6

def voc_ap(rec, prec, use_07_metric=False):
    if False:
        while True:
            i = 10
    ' ap = voc_ap(rec, prec, [use_07_metric])\n    Compute VOC AP given precision and recall.\n    If use_07_metric is true, uses the\n    VOC 07 11 point method (default:False).\n    '
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(all_boxes, all_gt_boxes, classes, ovthresh=0.5, use_07_metric=False):
    if False:
        return 10
    num_classes = len(classes)
    det_bbox = np.array([b[:5] for idx in all_boxes for b in idx])
    det_cls_idx = np.array([b[-1] for idx in all_boxes for b in idx])
    det_img_idx = np.array([img_idx for (img_idx, idx) in enumerate(all_boxes) for b in idx])
    MAP = np.zeros((num_classes, 1))
    det_bbox[:, :4] = np.round(det_bbox[:, :4], decimals=1)
    det_bbox[:, -1] = np.round(det_bbox[:, -1], decimals=3)
    for (cls_idx, cls) in enumerate(classes):
        if cls == '__background__':
            continue
        npos = 0
        for gt_boxes in all_gt_boxes:
            npos = npos + len(np.where(np.logical_and(gt_boxes[:, GT_CLASS_INDEX] == cls_idx, gt_boxes[:, GT_DIFFICULT_INDEX] == 0))[0])
        idx = np.where(det_cls_idx == cls_idx)[0]
        cls_bb = det_bbox[idx]
        cls_img_idx = det_img_idx[idx]
        sorted_ind = np.argsort(-cls_bb[:, -1])
        cls_bb = cls_bb[sorted_ind, :]
        cls_img_idx = cls_img_idx[sorted_ind]
        nd = len(sorted_ind)
        tp = np.zeros(len(sorted_ind))
        fp = np.zeros(len(sorted_ind))
        for d in range(nd):
            bb = cls_bb[d, :]
            img_idx = cls_img_idx[d]
            bbgt = all_gt_boxes[img_idx]
            box_idx = np.where(bbgt[:, GT_CLASS_INDEX] == cls_idx)[0]
            BBGT = bbgt[box_idx, :]
            ovmax = -np.inf
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih
                uni = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not bbgt[box_idx[jmax], GT_DIFFICULT_INDEX]:
                    if not bbgt[box_idx[jmax], GT_DETECTED_INDEX]:
                        tp[d] = 1.0
                        bbgt[box_idx[jmax], GT_DETECTED_INDEX] = True
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / (tp + fp + 1e-10)
        ap = voc_ap(rec, prec, True)
        MAP[cls_idx] = ap
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(MAP[1:].mean()))
    return MAP[1:]
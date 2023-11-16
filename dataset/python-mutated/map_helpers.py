import numpy as np
from utils.nms_wrapper import apply_nms_to_test_set_results

def evaluate_detections(all_boxes, all_gt_infos, classes, use_gpu_nms, device_id, apply_mms=True, nms_threshold=0.5, conf_threshold=0.0, use_07_metric=False):
    if False:
        return 10
    "\n    Computes per-class average precision.\n\n    Args:\n        all_boxes:          shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score\n        all_gt_infos:       a dictionary that contains all ground truth annoations in the following form:\n                            {'class_A': [{'bbox': array([[ 376.,  210.,  456.,  288.,   10.]], dtype=float32), 'det': [False], 'difficult': [False]}, ... ]}\n                            'class_B': [ <bbox_list> ], <more_class_to_bbox_list_entries> }\n        classes:            a list of class name, e.g. ['__background__', 'avocado', 'orange', 'butter']\n        use_07_metric:      whether to use VOC07's 11 point AP computation (default False)\n        apply_mms:          whether to apply non maximum suppression before computing average precision values\n        nms_threshold:      the threshold for discarding overlapping ROIs in nms\n        conf_threshold:     a minimum value for the score of an ROI. ROIs with lower score will be discarded\n\n    Returns:\n        aps - average precision value per class in a dictionary {classname: ap}\n    "
    if apply_mms:
        print('Number of rois before non-maximum suppression: %d' % sum([len(all_boxes[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
        (nms_dets, _) = apply_nms_to_test_set_results(all_boxes, nms_threshold, conf_threshold, use_gpu_nms, device_id)
        print('Number of rois  after non-maximum suppression: %d' % sum([len(nms_dets[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
    else:
        print('Skipping non-maximum suppression')
        nms_dets = all_boxes
    aps = {}
    for (classIndex, className) in enumerate(classes):
        if className != '__background__':
            (rec, prec, ap) = _evaluate_detections(classIndex, nms_dets, all_gt_infos[className], use_07_metric=use_07_metric)
            aps[className] = ap
    return aps

def _evaluate_detections(classIndex, all_boxes, gtInfos, overlapThreshold=0.5, use_07_metric=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Top level function that does the PASCAL VOC evaluation.\n    '
    num_images = len(all_boxes[0])
    detBboxes = []
    detImgIndices = []
    detConfidences = []
    for imgIndex in range(num_images):
        dets = all_boxes[classIndex][imgIndex]
        if dets != []:
            for k in range(dets.shape[0]):
                detImgIndices.append(imgIndex)
                detConfidences.append(dets[k, -1])
                detBboxes.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
    detBboxes = np.array(detBboxes)
    detConfidences = np.array(detConfidences)
    (rec, prec, ap) = _voc_computePrecisionRecallAp(class_recs=gtInfos, confidence=detConfidences, image_ids=detImgIndices, BB=detBboxes, ovthresh=overlapThreshold, use_07_metric=use_07_metric)
    return (rec, prec, ap)

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    if False:
        i = 10
        return i + 15
    '\n    Computes VOC AP given precision and recall.\n    '
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.0
    else:
        mrecalls = np.concatenate(([0.0], recalls, [1.0]))
        mprecisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap

def _voc_computePrecisionRecallAp(class_recs, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes precision, recall. and average precision\n    '
    if len(BB) == 0:
        return (0.0, 0.0, 0.0)
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
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
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0
    npos = sum([len(cr['bbox']) for cr in class_recs])
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = computeAveragePrecision(rec, prec, use_07_metric)
    return (rec, prec, ap)
import numpy as np
from utils.cython_modules.cpu_nms import cpu_nms
try:
    from utils.cython_modules.gpu_nms import gpu_nms
    gpu_nms_available = True
except ImportError:
    gpu_nms_available = False

def nms(dets, thresh, use_gpu_nms=True, device_id=0):
    if False:
        i = 10
        return i + 15
    '\n    Dispatches the call to either CPU or GPU NMS implementations\n    '
    if dets.shape[0] == 0:
        return []
    if gpu_nms_available and use_gpu_nms:
        return gpu_nms(dets, thresh, device_id=device_id)
    else:
        return cpu_nms(dets, thresh)

def apply_nms_to_single_image_results(coords, labels, scores, use_gpu_nms, device_id, nms_threshold=0.5, conf_threshold=0.0):
    if False:
        i = 10
        return i + 15
    '\n    Applies nms to the results for a single image.\n\n    Args:\n        coords:             (x_min, y_min, x_max, y_max) coordinates for n rois. shape = (n, 4)\n        labels:             the predicted label per roi. shape = (n, 1)\n        scores:             the predicted score per roi. shape = (n, 1)\n        nms_threshold:      the threshold for discarding overlapping ROIs in nms\n        conf_threshold:     a minimum value for the score of an ROI. ROIs with lower score will be discarded\n\n    Returns:\n        nmsKeepIndices - the indices of the ROIs to keep after nms\n    '
    allIndices = []
    nmsRects = [[[]] for _ in range(max(labels) + 1)]
    coordsWithScores = np.hstack((coords, np.array([scores]).T))
    for i in range(max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        nmsRects[i][0] = coordsWithScores[indices, :]
        allIndices.append(indices)
    (_, nmsKeepIndicesList) = apply_nms_to_test_set_results(nmsRects, nms_threshold, conf_threshold, use_gpu_nms, device_id)
    nmsKeepIndices = []
    for i in range(max(labels) + 1):
        for keepIndex in nmsKeepIndicesList[i][0]:
            nmsKeepIndices.append(allIndices[i][keepIndex])
    assert len(nmsKeepIndices) == len(set(nmsKeepIndices))
    return nmsKeepIndices

def apply_nms_to_test_set_results(all_boxes, nms_threshold, conf_threshold, use_gpu_nms, device_id):
    if False:
        i = 10
        return i + 15
    '\n    Applies nms to the results of multiple images.\n\n    Args:\n        all_boxes:      shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score\n        nms_threshold:  the threshold for discarding overlapping ROIs in nms\n        conf_threshold: a minimum value for the score of an ROI. ROIs with lower score will be discarded\n\n    Returns:\n        nms_boxes - the reduced set of rois after nms\n        nmsKeepIndices - the indices of the ROIs to keep after nms\n    '
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    nms_keepIndices = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if len(dets) == 0:
                continue
            if len(dets) == 1:
                keep = [0]
            else:
                keep = nms(dets.astype(np.float32), nms_threshold, use_gpu_nms, device_id)
            if conf_threshold > 0:
                keep_conf_idx = np.where(dets[:, -1] > conf_threshold)
                keep = list(set(keep_conf_idx[0]).intersection(keep))
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_keepIndices[cls_ind][im_ind] = keep
    return (nms_boxes, nms_keepIndices)
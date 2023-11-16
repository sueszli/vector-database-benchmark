from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from .image import transform_preds

def ctdet_post_process(dets, c, s, h, w, num_classes):
    if False:
        i = 10
        return i + 15
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32), dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret
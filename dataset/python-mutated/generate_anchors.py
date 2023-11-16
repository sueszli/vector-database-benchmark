from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np

def generate_all_anchors(conv_size_x, conv_size_y, im_scale, scales=np.array((8, 16, 32))):
    if False:
        print('Hello World!')
    anchors = generate_anchors(scales=scales)
    num_anchors = anchors.shape[0]
    shift_x = np.arange(0, conv_size_x) * 1.0 / im_scale
    shift_y = np.arange(0, conv_size_y) * 1.0 / im_scale
    (shift_x, shift_y) = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = num_anchors
    A = shifts.shape[0]
    all_anchors = anchors.reshape((1, K, 4)).transpose((1, 0, 2)) + shifts.reshape((1, A, 4))
    all_anchors = all_anchors.reshape((A * K, 4))
    return all_anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    if False:
        i = 10
        return i + 15
    '\n    Generate anchor (reference) windows by enumerating aspect ratios X\n    scales wrt a reference (0, 0, 15, 15) window.\n    '
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return width, height, x center, and y center for an anchor (window).\n    '
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return (w, h, x_ctr, y_ctr)

def _mkanchors(ws, hs, x_ctr, y_ctr):
    if False:
        print('Hello World!')
    '\n    Given a vector of widths (ws) and heights (hs) around a center\n    (x_ctr, y_ctr), output a set of anchors (windows).\n    '
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    if False:
        i = 10
        return i + 15
    '\n    Enumerate a set of anchors for each aspect ratio wrt an anchor.\n    '
    (w, h, x_ctr, y_ctr) = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    if False:
        while True:
            i = 10
    '\n    Enumerate a set of anchors for each scale wrt an anchor.\n    '
    (w, h, x_ctr, y_ctr) = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed
    embed()
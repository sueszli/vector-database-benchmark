import numpy as np
import cntk as C

def SmoothL1Loss(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    if False:
        while True:
            i = 10
    '\n        From https://github.com/smallcorgi/Faster-RCNN_TF/blob/master/lib/fast_rcnn/train.py\n\n        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))\n        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2\n                        |x| - 0.5 / sigma^2,    otherwise\n    '
    sigma2 = sigma * sigma
    inside_mul_abs = C.abs(C.element_times(bbox_inside_weights, C.minus(bbox_pred, bbox_targets)))
    smooth_l1_sign = C.less(inside_mul_abs, 1.0 / sigma2)
    smooth_l1_option1 = C.element_times(C.element_times(inside_mul_abs, inside_mul_abs), 0.5 * sigma2)
    smooth_l1_option2 = C.minus(inside_mul_abs, 0.5 / sigma2)
    smooth_l1_result = C.plus(C.element_times(smooth_l1_option1, smooth_l1_sign), C.element_times(smooth_l1_option2, C.minus(1.0, smooth_l1_sign)))
    return C.element_times(bbox_outside_weights, smooth_l1_result)
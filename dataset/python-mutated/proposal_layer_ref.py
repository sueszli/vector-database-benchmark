from __future__ import print_function
from builtins import object
import numpy as np
from generate_anchors import generate_anchors
from util import bbox_transform_inv, clip_boxes, nms
DEBUG = False

class PyCaffeProposalLayer(object):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top, pre_nms_topN=12000, post_nms_topN=2000, nms_thresh=0.7, min_size=16):
        if False:
            return 10
        self._feat_stride = 16
        anchor_scales = (8, 16, 32)
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self.pre_nms_topN = pre_nms_topN
        self.post_nms_topN = post_nms_topN
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if DEBUG:
            print('feat_stride: {}'.format(self._feat_stride))
            print('anchors:')
            print(self._anchors)

    def forward(self, bottom, top):
        if False:
            i = 10
            return i + 15
        assert bottom[0].shape[0] == 1, 'Only single item batches are supported'
        scores = bottom[0][:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1]
        im_info = [float(x.get()) for x in bottom[2]]
        if DEBUG:
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))
        (height, width) = scores.shape[-2:]
        if DEBUG:
            print('score map size: {}'.format(scores.shape))
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        (shift_x, shift_y) = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)).transpose((1, 0, 2)) + shifts.reshape((1, K, 4))
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])
        keep = _filter_boxes(proposals, self.min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
        if DEBUG:
            print('(CAFFE) len(keep) before nms: {}'.format(len(keep)))
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        if DEBUG:
            print('(CAFFE) len(proposals) after get_top_N: {}'.format(len(proposals)))
        keep = nms(np.hstack((proposals, scores)), self.nms_thresh)
        if DEBUG:
            print('(CAFFE) len(keep) before clipping: {}'.format(len(keep)))
        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        if DEBUG:
            print('(CAFFE) len(keep) after nms: {}'.format(len(keep)))
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0] = blob
        top[1] = scores

    def backward(self, top, propagate_down, bottom):
        if False:
            for i in range(10):
                print('nop')
        'This layer does not propagate gradients.'
        pass

    def reshape(self, bottom, top):
        if False:
            i = 10
            return i + 15
        'Reshaping happens during the call to forward.'
        pass

def _filter_boxes(boxes, min_size):
    if False:
        return 10
    'Remove all boxes with any side smaller than min_size.'
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
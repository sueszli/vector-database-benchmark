import os
import yaml
import numpy as np
import numpy.random as npr
from utils.caffe_layers.default_config import cfg
from utils.rpn.generate_anchors import generate_anchors
from utils.rpn.bbox_transform import bbox_transform
from utils.cython_modules.cython_bbox import bbox_overlaps
DEBUG = False

class AnchorTargetLayer:
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def set_param_str(self, param_str):
        if False:
            while True:
                i = 10
        self.param_str_ = param_str

    def set_deterministic_mode(self, mode=True):
        if False:
            return 10
        self._determininistic_mode = mode

    def setup(self, bottom, top):
        if False:
            while True:
                i = 10
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']
        if DEBUG:
            print('anchors:')
            print(self._anchors)
            print('anchor shapes:')
            print(np.hstack((self._anchors[:, 2::4] - self._anchors[:, 0::4], self._anchors[:, 3::4] - self._anchors[:, 1::4])))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0
        self._allowed_border = layer_params.get('allowed_border', 0)
        (height, width) = bottom[0].data.shape[-2:]
        if DEBUG:
            print('AnchorTargetLayer: height', height, 'width', width)

    def forward(self, bottom, top):
        if False:
            return 10
        assert bottom[0].data.shape[0] == 1, 'Only single item batches are supported'
        (height, width) = bottom[0].data.shape[-2:]
        gt_boxes = bottom[1]
        im_info = bottom[2]
        if DEBUG:
            print('')
            print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print('scale: {}'.format(im_info[2]))
            print('height, width: ({}, {})'.format(height, width))
            print('rpn: gt_boxes.shape', gt_boxes.shape)
            print('rpn: gt_boxes', gt_boxes)
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        (shift_x, shift_y) = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        inds_inside = np.where((all_anchors[:, 0] >= -self._allowed_border) & (all_anchors[:, 1] >= -self._allowed_border) & (all_anchors[:, 2] < im_info[1] + self._allowed_border) & (all_anchors[:, 3] < im_info[0] + self._allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print('anchors.shape', anchors.shape)
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)
        overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            if self._determininistic_mode:
                disable_inds = fg_inds[:len(fg_inds) - num_fg]
            else:
                disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
            labels[disable_inds] = -1
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            if self._determininistic_mode:
                disable_inds = bg_inds[:len(bg_inds) - num_bg]
            else:
                disable_inds = npr.choice(bg_inds, size=len(bg_inds) - num_bg, replace=False)
            labels[disable_inds] = -1
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
            negative_weights = (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
        if DEBUG:
            print('rpn: max max_overlap', np.max(max_overlaps))
            print('rpn: num_positive', np.sum(labels == 1))
            print('rpn: num_negative', np.sum(labels == 0))
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print('rpn: num_positive avg', self._fg_sum / self._count)
            print('rpn: num_negative avg', self._bg_sum / self._count)
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        return (labels, bbox_targets, bbox_inside_weights)

    def backward(self, top, propagate_down, bottom):
        if False:
            return 10
        'This layer does not propagate gradients.'
        pass

    def reshape(self, bottom, top):
        if False:
            i = 10
            return i + 15
        'Reshaping happens during the call to forward.'
        pass

def _unmap(data, count, inds, fill=0):
    if False:
        return 10
    ' Unmap a subset of item (data) back to the original set of items (of\n    size count) '
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_targets(ex_rois, gt_rois):
    if False:
        print('Hello World!')
    'Compute bounding-box regression targets for an image.'
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
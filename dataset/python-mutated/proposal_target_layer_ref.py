from __future__ import division
from __future__ import print_function
from builtins import object
import numpy as np
import numpy.random as npr
from util import bbox_transform, calculate_bb_overlap
DEBUG = False
FG_THRESH = 0.5
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0
FG_FRAC = 0.25
BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
BBOX_NORMALIZE_MEANS = [0.0, 0.0, 0.0, 0.0]
BBOX_NORMALIZE_STDS = [0.1, 0.1, 0.2, 0.2]

class PyCaffeProposalTargetLayer(object):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top, deterministic=False):
        if False:
            for i in range(10):
                print('nop')
        self._num_classes = 21
        self.deterministic = deterministic

    def forward(self, bottom, top):
        if False:
            return 10
        all_rois = bottom[0]
        gt_boxes = bottom[1]
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'
        rois_per_image = 128
        fg_rois_per_image = np.round(FG_FRAC * rois_per_image)
        (labels, rois, bbox_targets, bbox_inside_weights) = _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, self._num_classes, self.deterministic)
        if DEBUG:
            print('num fg: {}'.format((labels > 0).sum()))
            print('num bg: {}'.format((labels == 0).sum()))
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print('num fg avg: {}'.format(self._fg_num / self._count))
            print('num bg avg: {}'.format(self._bg_num / self._count))
            print('ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num)))
        top[1] = labels
        top[2] = bbox_targets
        top[3] = bbox_inside_weights

    def backward(self, top, propagate_down, bottom):
        if False:
            return 10
        'This layer does not propagate gradients.'
        pass

    def reshape(self, bottom, top):
        if False:
            for i in range(10):
                print('nop')
        'Reshaping happens during the call to forward.'
        pass

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    if False:
        while True:
            i = 10
    'Bounding-box regression targets (bbox_target_data) are stored in a\n    compact form N x (class, tx, ty, tw, th)\n    This function expands those targets into the 4-of-4*K representation used\n    by the network (i.e. only one class has non-zero targets).\n    Returns:\n        bbox_target (ndarray): N x 4K blob of regression targets\n        bbox_inside_weights (ndarray): N x 4K blob of loss weights\n    '
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = 1
    return (bbox_targets, bbox_inside_weights)

def _compute_targets(ex_rois, gt_rois, labels):
    if False:
        while True:
            i = 10
    'Compute bounding-box regression targets for an image.'
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    targets = bbox_transform(ex_rois, gt_rois)
    if BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = (targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS)
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, deterministic=False):
    if False:
        return 10
    'Generate a random sample of RoIs comprising foreground and background\n    examples.\n    '
    overlaps = calculate_bb_overlap(all_rois[:, 1:], gt_boxes[:, :4])
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    if fg_inds.size > 0 and (not deterministic):
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    elif fg_inds.size > 0:
        fg_inds = fg_inds[:fg_rois_per_this_image]
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) & (max_overlaps >= BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    if bg_inds.size > 0 and (not deterministic):
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    elif bg_inds.size > 0:
        bg_inds = bg_inds[:bg_rois_per_this_image]
    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds]
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    (bbox_targets, bbox_inside_weights) = _get_bbox_regression_labels(bbox_target_data, num_classes)
    return (labels, rois, bbox_targets, bbox_inside_weights)
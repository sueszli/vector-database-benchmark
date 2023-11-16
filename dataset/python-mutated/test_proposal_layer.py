"""
Test for proposal layer and proposal target layer. It uses the layer implementations from
faster rcnn code (https://github.com/rbgirshick/py-faster-rcnn) and currently in the file
proposal_layer_ref.py
proposal_target_layer_ref.py

"""
from __future__ import division
from builtins import range
from builtins import object
import numpy as np
import itertools as itt
from neon.backends import gen_backend
from proposal_layer_ref import PyCaffeProposalLayer
from proposal_target_layer_ref import PyCaffeProposalTargetLayer
from proposal_layer import ProposalLayer

def pytest_generate_tests(metafunc):
    if False:
        for i in range(10):
            print('nop')
    if 'fargs' in metafunc.fixturenames:
        if metafunc.config.option.all:
            _conv_size = [62, 63]
            im_shape = [[800, 600], [460, 1000]]
            SCALE = [1.6, 2.0]
            pre_nms_topN = [12000, 6000]
            post_nms_topN = [2000, 300]
            nms_thresh = [0.7]
            min_size = [16]
        else:
            _conv_size = [63]
            im_shape = [[800, 600]]
            SCALE = [1.6, 2.5]
            pre_nms_topN = [12000]
            post_nms_topN = [2000]
            nms_thresh = [0.7]
            min_size = [16]
        fargs = itt.product(_conv_size, im_shape, SCALE, pre_nms_topN, post_nms_topN, nms_thresh, min_size)
        metafunc.parametrize('fargs', fargs)

class mock_layer(object):

    def __init__(self, out):
        if False:
            while True:
                i = 10
        self.outputs = out
        self.out_shape = (1, 1)
        self.shape = (1, 1)
        self.parallelism = False

class mock_dataloader(object):

    def __init__(self, conv_size, conv_scale, im_shape, SCALE, gt_boxes, gt_classes, num_gt_boxes, frcn_labels, frcn_labels_mask, frcn_bbtargets, frcn_bbmask):
        if False:
            print('Hello World!')
        self.im_shape = im_shape
        self.im_scale = SCALE
        self.gt_boxes = gt_boxes
        self.gt_classes = gt_classes
        self.num_gt_boxes = num_gt_boxes
        self.frcn_labels = frcn_labels
        self.frcn_labels_mask = frcn_labels_mask
        self.frcn_bbtargets = frcn_bbtargets
        self.frcn_bbmask = frcn_bbmask
        self.conv_height = conv_size
        self.conv_width = conv_size
        self.conv_scale = conv_scale.get()
        self.num_classes = 21

    def get_metadata_buffers(self):
        if False:
            i = 10
            return i + 15
        return (self.im_shape, self.im_scale, self.gt_boxes, self.gt_classes, self.num_gt_boxes, None)

    def get_target_buffers(self):
        if False:
            i = 10
            return i + 15
        return ((self.frcn_labels, self.frcn_labels_mask), (self.frcn_bbtargets, self.frcn_bbmask))

def test_proposal_layer(backend_default, fargs):
    if False:
        return 10
    np.random.seed(seed=0)
    be = backend_default
    be.bsz = 1
    (_conv_size, im_shape_arr, SCALE, pre_nms_topN, post_nms_topN, nms_thresh, min_size) = fargs
    im_shape = be.zeros((2, 1), dtype=np.float32)
    im_shape[:] = np.array(im_shape_arr)
    im_scale = be.ones((1, 1), dtype=np.float32).fill(1.0 / 16.0)
    SCALE = be.ones((1, 1), dtype=np.float32).fill(SCALE)
    real_H = np.round(im_shape.get()[1] * im_scale.get()).astype(int).reshape((1,))[0]
    real_W = np.round(im_shape.get()[0] * im_scale.get()).astype(int).reshape((1,))[0]
    frcn_labels = be.zeros((21, 128), dtype=np.int32)
    frcn_labels_mask = be.zeros(frcn_labels.shape, dtype=np.int32)
    frcn_bbtargets = be.zeros((21 * 4, 128), dtype=np.float32)
    frcn_bbmask = be.zeros(frcn_bbtargets.shape, dtype=np.float32)
    gt_boxes = be.zeros((64, 4), dtype=np.float32)
    gt_boxes[:3, :] = np.array([[262, 210, 323, 338], [164, 263, 252, 371], [240, 193, 294, 298]])
    gt_classes = be.zeros((64, 1), dtype=np.int32)
    gt_classes[:3, :] = np.array([[9], [9], [9]])
    num_gt_boxes = be.zeros((1, 1), dtype=np.int32).fill(3)
    num_scores = 2 * 9 * _conv_size * _conv_size
    rpn_obj_scores_dev = be.array(np.random.choice(num_scores * 2, size=num_scores, replace=False) / float(num_scores * 2.0))
    rpn_bbox_deltas_dev = be.array(np.random.random((4 * 9 * _conv_size * _conv_size, 1)))
    RPN_1x1_obj = mock_layer(rpn_obj_scores_dev)
    RPN_1x1_bbox = mock_layer(rpn_bbox_deltas_dev)
    mock_loader = mock_dataloader(_conv_size, im_scale, im_shape, SCALE, gt_boxes, gt_classes, num_gt_boxes, frcn_labels, frcn_labels_mask, frcn_bbtargets, frcn_bbmask)
    prop_layer = ProposalLayer([[RPN_1x1_obj], [RPN_1x1_bbox]], mock_loader, pre_nms_N=pre_nms_topN, post_nms_N=post_nms_topN, nms_thresh=nms_thresh, min_bbox_size=min_size, num_rois=128, deterministic=True, inference=False, debug=True)
    prop_layer.configure(mock_layer([]))
    prop_layer.allocate()
    inputs = []
    (inputs, dev_proposals) = prop_layer.fprop(inputs, inference=False)
    target_proposals = prop_layer.proposals
    target_scores = prop_layer.scores
    prop_layer_ref = PyCaffeProposalLayer()
    rpn_obj_scores = rpn_obj_scores_dev.get()
    rpn_bbox_deltas = rpn_bbox_deltas_dev.get()
    rpn_bbox_deltas = rpn_bbox_deltas.reshape((4, -1, _conv_size, _conv_size))
    rpn_bbox_deltas = rpn_bbox_deltas[:, :, :real_H, :real_W].reshape((4, -1)).T
    rpn_obj_scores = rpn_obj_scores.reshape((2, -1, _conv_size, _conv_size))
    rpn_obj_scores = rpn_obj_scores[:, :, :real_H, :real_W].transpose((0, 1, 2, 3))
    rpn_obj_scores = rpn_obj_scores.reshape((1, -1, real_H, real_W))
    bottom = [None, None, None]
    bottom[0] = rpn_obj_scores
    bottom[1] = rpn_bbox_deltas
    bottom[2] = [im_shape[1], im_shape[0], SCALE]
    top = [None, None]
    prop_layer_ref.setup(bottom, top, pre_nms_topN=pre_nms_topN, post_nms_topN=post_nms_topN, nms_thresh=nms_thresh, min_size=min_size)
    prop_layer_ref.forward(bottom, top)
    assert np.allclose(top[0][:, 1:], target_proposals, atol=1e-05, rtol=0.0001)
    assert np.allclose(top[1], target_scores, atol=1e-05, rtol=0.0001)
    t_bottom = [0, 1]
    zeros = np.zeros((target_proposals.shape[0], 1), dtype=target_proposals.dtype)
    t_bottom[0] = np.hstack((zeros, target_proposals))
    t_bottom[1] = np.hstack((prop_layer.gt_boxes.get(), prop_layer.gt_classes.get()))[:prop_layer.num_gt_boxes.get()[0][0]]
    t_top = [None, None, None, None, None]
    prop_target_layer_ref = PyCaffeProposalTargetLayer()
    prop_target_layer_ref.setup(t_bottom, t_top, deterministic=True)
    prop_target_layer_ref.forward(t_bottom, t_top)
    frcn_bbtargets_reference = np.zeros(frcn_bbtargets.shape, dtype=np.float32)
    frcn_bbmask_reference = np.zeros(frcn_bbmask.shape, dtype=np.float32)
    frcn_bbtargets_reference[:t_top[2].shape[0]] = t_top[2].T
    frcn_bbmask_reference[:t_top[3].shape[0]] = t_top[3].T
    neon_labels = np.zeros((frcn_labels.shape[1],))
    label_mat = frcn_labels.get() * frcn_labels_mask.get()
    for cls in range(frcn_labels.shape[0]):
        for (idx, elem) in enumerate(label_mat[cls]):
            if elem != 0:
                neon_labels[idx] = cls
    assert np.alltrue(t_top[1] == neon_labels)
    assert np.allclose(frcn_bbtargets_reference, frcn_bbtargets.get(), atol=0.0001)
    assert np.alltrue(frcn_bbmask_reference == frcn_bbmask.get())
if __name__ == '__main__':
    be = gen_backend(backend='gpu', batch_size=1)
    _conv_size = 62
    im_shape = [800, 600]
    SCALE = 1.6
    pre_nms_topN = 12000
    post_nms_topN = 2000
    nms_thresh = 0.7
    min_size = 16
    fargs = (_conv_size, im_shape, SCALE, pre_nms_topN, post_nms_topN, nms_thresh, min_size)
    test_proposal_layer(be, fargs)
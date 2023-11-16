import unittest
import numpy as np
from op_test import OpTest
from test_anchor_generator_op import anchor_generator_in_python
from test_generate_proposals_op import box_coder, clip_tiled_boxes, nms
import paddle

def python_generate_proposals_v2(scores, bbox_deltas, img_size, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, pixel_offset=False, return_rois_num=True):
    if False:
        for i in range(10):
            print('nop')
    (rpn_rois, rpn_roi_probs, rpn_rois_num) = paddle.vision.ops.generate_proposals(scores, bbox_deltas, img_size, anchors, variances, pre_nms_top_n=pre_nms_top_n, post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=min_size, eta=eta, pixel_offset=pixel_offset, return_rois_num=return_rois_num)
    return (rpn_rois, rpn_roi_probs)

def generate_proposals_v2_in_python(scores, bbox_deltas, im_shape, anchors, variances, pre_nms_topN, post_nms_topN, nms_thresh, min_size, eta, pixel_offset):
    if False:
        return 10
    all_anchors = anchors.reshape(-1, 4)
    rois = np.empty((0, 5), dtype=np.float32)
    roi_probs = np.empty((0, 1), dtype=np.float32)
    rpn_rois = []
    rpn_roi_probs = []
    rois_num = []
    num_images = scores.shape[0]
    for img_idx in range(num_images):
        (img_i_boxes, img_i_probs) = proposal_for_one_image(im_shape[img_idx, :], all_anchors, variances, bbox_deltas[img_idx, :, :, :], scores[img_idx, :, :, :], pre_nms_topN, post_nms_topN, nms_thresh, min_size, eta, pixel_offset)
        rois_num.append(img_i_probs.shape[0])
        rpn_rois.append(img_i_boxes)
        rpn_roi_probs.append(img_i_probs)
    return (rpn_rois, rpn_roi_probs, rois_num)

def proposal_for_one_image(im_shape, all_anchors, variances, bbox_deltas, scores, pre_nms_topN, post_nms_topN, nms_thresh, min_size, eta, pixel_offset):
    if False:
        while True:
            i = 10
    bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape(-1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    variances = variances.reshape(-1, 4)
    scores = scores.transpose((1, 2, 0)).reshape(-1, 1)
    if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
        order = np.argsort(-scores.squeeze())
    else:
        inds = np.argpartition(-scores.squeeze(), pre_nms_topN)[:pre_nms_topN]
        order = np.argsort(-scores[inds].squeeze())
        order = inds[order]
    scores = scores[order, :]
    bbox_deltas = bbox_deltas[order, :]
    all_anchors = all_anchors[order, :]
    proposals = box_coder(all_anchors, bbox_deltas, variances, pixel_offset)
    proposals = clip_tiled_boxes(proposals, im_shape, pixel_offset)
    keep = filter_boxes(proposals, min_size, im_shape, pixel_offset)
    if len(keep) == 0:
        proposals = np.zeros((1, 4)).astype('float32')
        scores = np.zeros((1, 1)).astype('float32')
        return (proposals, scores)
    proposals = proposals[keep, :]
    scores = scores[keep, :]
    if nms_thresh > 0:
        keep = nms(boxes=proposals, scores=scores, nms_threshold=nms_thresh, eta=eta, pixel_offset=pixel_offset)
        if post_nms_topN > 0 and post_nms_topN < len(keep):
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep, :]
    return (proposals, scores)

def filter_boxes(boxes, min_size, im_shape, pixel_offset=True):
    if False:
        i = 10
        return i + 15
    'Only keep boxes with both sides >= min_size and center within the image.'
    min_size = max(min_size, 1.0)
    offset = 1 if pixel_offset else 0
    ws = boxes[:, 2] - boxes[:, 0] + offset
    hs = boxes[:, 3] - boxes[:, 1] + offset
    if pixel_offset:
        x_ctr = boxes[:, 0] + ws / 2.0
        y_ctr = boxes[:, 1] + hs / 2.0
        keep = np.where((ws >= min_size) & (hs >= min_size) & (x_ctr < im_shape[1]) & (y_ctr < im_shape[0]))[0]
    else:
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

class TestGenerateProposalsV2Op(OpTest):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {'Scores': self.scores, 'BboxDeltas': self.bbox_deltas, 'ImShape': self.im_shape.astype(np.float32), 'Anchors': self.anchors, 'Variances': self.variances}
        self.attrs = {'pre_nms_topN': self.pre_nms_topN, 'post_nms_topN': self.post_nms_topN, 'nms_thresh': self.nms_thresh, 'min_size': self.min_size, 'eta': self.eta, 'pixel_offset': self.pixel_offset}
        self.outputs = {'RpnRois': self.rpn_rois[0], 'RpnRoiProbs': self.rpn_roi_probs[0]}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'generate_proposals_v2'
        self.python_api = python_generate_proposals_v2
        self.python_out_sig = ['Out']
        self.set_data()

    def init_test_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.pre_nms_topN = 12000
        self.post_nms_topN = 5000
        self.nms_thresh = 0.7
        self.min_size = 3.0
        self.eta = 1.0
        self.pixel_offset = True

    def init_test_input(self):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 1
        input_channels = 20
        layer_h = 16
        layer_w = 16
        input_feat = np.random.random((batch_size, input_channels, layer_h, layer_w)).astype('float32')
        (self.anchors, self.variances) = anchor_generator_in_python(input_feat=input_feat, anchor_sizes=[16.0, 32.0], aspect_ratios=[0.5, 1.0], variances=[1.0, 1.0, 1.0, 1.0], stride=[16.0, 16.0], offset=0.5)
        self.im_shape = np.array([[64, 64]]).astype('float32')
        num_anchors = self.anchors.shape[2]
        self.scores = np.random.random((batch_size, num_anchors, layer_h, layer_w)).astype('float32')
        self.bbox_deltas = np.random.random((batch_size, num_anchors * 4, layer_h, layer_w)).astype('float32')

    def init_test_output(self):
        if False:
            return 10
        (self.rpn_rois, self.rpn_roi_probs, self.rois_num) = generate_proposals_v2_in_python(self.scores, self.bbox_deltas, self.im_shape, self.anchors, self.variances, self.pre_nms_topN, self.post_nms_topN, self.nms_thresh, self.min_size, self.eta, self.pixel_offset)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
"""
Define a layer that takes Region Proposal Network's output and generate
region proposals in the format of bounding boxes, compares with ground truth boxes
and generates bounding box target labels and regression targets
"""
from __future__ import division
import numpy as np
from neon.layers.layer import Layer
from generate_anchors import generate_all_anchors
from util import compute_targets, calculate_bb_overlap
FG_THRESH = 0.5
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0
FG_FRACTION = 0.25
BBOX_NORMALIZE_MEANS = [0.0, 0.0, 0.0, 0.0]
BBOX_NORMALIZE_STDS = [0.1, 0.1, 0.2, 0.2]

class ProposalLayer(Layer):
    """
    Proposal layer takes as input:
    (1) output of RPN cls_score
    (2) output of RPN bbox regression

    Converts that to an output ROIs
    (5, num_ROIs) <- [image_idx, x_min, y_min, x_max, y_max]

    Steps:
    1. For each anchor, generate anchor boxes and apply bbox deltas to get
       bbox proposals
    2. Clip bboxes to image
    3. remove bbox with H < threshold or W < threshold
    4. set the scores to be -1 for the padded area
    5. Take top pre_nms_N scores
    6. apply NMS with a threshold
    7. return the top num_ROIs proposals
    8. provide ROIs
    9. compute bbox targets and store in target_buffers


    """

    def __init__(self, rpn_layers, dataloader, inference=False, num_rois=128, pre_nms_N=12000, post_nms_N=2000, nms_thresh=0.7, min_bbox_size=16, fg_fraction=None, fg_thresh=None, bg_thresh_hi=None, bg_thresh_lo=None, deterministic=False, name=None, debug=False):
        if False:
            return 10
        '\n        Arguments:\n            rpn_layers (list): References to the RPN layers: [RPN_1x1_obj, RPN_1x1_bbox]\n            target_buffers (tuple): Target buffers for training fast-rcnn: (class, bbox regression)\n            num_rois (int, optional): Number of ROIs to sample from proposed (default: 128)\n            pre_nms_N (int, optional): Number of ROIs to retain before using NMS (default: 12000)\n            post_nms_N (int, optional): Number of ROIs to retain after using NMS (default: 2000)\n            nms_thresh (float, optional): Threshold for non-maximum supression (default: 0.7)\n            min_bbox_size (integer, optional): Minimize bboxes side length (default: 16)\n            name (string, optional): Name of layer (default: None)\n        '
        super(ProposalLayer, self).__init__(name)
        (self.rpn_obj, self.rpn_bbox) = rpn_layers
        self.num_rois = num_rois
        self.pre_nms_N = pre_nms_N
        self.post_nms_N = post_nms_N
        self.nms_thresh = nms_thresh
        self.min_bbox_size = min_bbox_size
        self.num_classes = dataloader.num_classes
        self.fg_fraction = fg_fraction if fg_fraction else FG_FRACTION
        self.fg_thresh = fg_thresh if fg_thresh else FG_THRESH
        self.bg_thresh_hi = bg_thresh_hi if bg_thresh_hi else BG_THRESH_HI
        self.bg_thresh_lo = bg_thresh_lo if bg_thresh_lo else BG_THRESH_LO
        self.deterministic = deterministic
        self.debug = debug
        self.inference = inference
        self.dataloader = dataloader
        self._conv_height = dataloader.conv_height
        self._conv_width = dataloader.conv_width
        self._scale = dataloader.conv_scale
        self._anchors = generate_all_anchors(self._conv_height, self._conv_width, self._scale)
        self._dev_anchors = self.be.array(self._anchors)
        self._num_anchors = self._anchors.shape[0]

    def configure(self, in_obj):
        if False:
            i = 10
            return i + 15
        super(ProposalLayer, self).configure(in_obj)
        if self.inference:
            self.out_shape = (5, self.post_nms_N)
        else:
            self.out_shape = (5, self.num_rois)
        self.in_shape = in_obj.out_shape
        return (in_obj, self)

    def get_description(self, **kwargs):
        if False:
            while True:
                i = 10
        skip = ['rpn_layers', 'global_buffers', 'dataloader']
        if 'skip' in kwargs:
            kwargs['skip'].append(skip)
        else:
            kwargs['skip'] = skip
        return super(ProposalLayer, self).get_description(**kwargs)

    def allocate(self):
        if False:
            return 10
        super(ProposalLayer, self).allocate()
        self._proposals = self.be.zeros((self._num_anchors, 4))
        self.dets = self.be.zeros((self.pre_nms_N, 5))
        self.dev_proposals = self.be.zeros((self.post_nms_N, 5))
        self.dev_proposals_filtered = self.be.zeros((self.num_rois, 5))
        self.rpn_scores_v = None
        self.bbox_deltas_v = None
        self._scores = self.be.zeros((self._num_anchors, 1))

    def fprop(self, inputs, inference=False):
        if False:
            while True:
                i = 10
        '\n        fprop function that does no proposal filtering\n        '
        assert self.inference == inference, 'Model was configured for inference={}'.format(self.inference)
        (self.im_shape, self.im_scale, self.gt_boxes, self.gt_classes, self.num_gt_boxes, _) = self.dataloader.get_metadata_buffers()
        real_H = int(np.round(self.im_shape.get()[1] * self._scale))
        real_W = int(np.round(self.im_shape.get()[0] * self._scale))
        if self.rpn_scores_v is None:
            self.rpn_scores_v = self.rpn_obj[0].outputs.reshape((2, -1))[1].T
        if self.bbox_deltas_v is None:
            self.bbox_deltas_v = self.rpn_bbox[0].outputs.reshape((4, -1)).T
        self._bbox_transform_inv(self._dev_anchors, self.bbox_deltas_v, output=self._proposals)
        self._clip_boxes(self._proposals, self.im_shape)
        keep = self._filter_boxes(self._proposals, self.min_bbox_size * float(self.im_scale.get()))
        self._scores[:] = self.rpn_scores_v * keep - (1 - keep)
        scores_np = self._scores.get()
        scores_np.reshape(-1, self._conv_height, self._conv_width)[:, real_H:] = -1
        scores_np.reshape(-1, self._conv_height, self._conv_width)[:, :, real_W:] = -1
        self._scores[:] = self.be.array(scores_np)
        top_N_ind = self.get_top_N_index(self._scores, self.pre_nms_N)
        self.dets.fill(0)
        self.dets[:len(top_N_ind), :4] = self._proposals[top_N_ind]
        self.dets[:len(top_N_ind), 4] = self._scores[top_N_ind]
        keep = self.be.nms(self.dets, self.nms_thresh)
        keep = keep[:self.post_nms_N]
        self.num_proposals = len(keep)
        if self.debug or not inference:
            self.proposals = self.dets[keep, :4].get()
            self.scores = self.dets[keep, -1].get()
        self.dev_proposals.fill(0)
        self.dev_proposals[:self.num_proposals, 1:] = self.dets[keep, :4]
        if not inference:
            ((frcn_labels, frcn_labels_mask), (frcn_bbtargets, frcn_bbmask)) = self.dataloader.get_target_buffers()
            non_zero_gt_boxes = self.gt_boxes.get()
            num_gt_boxes = self.num_gt_boxes.get()[0][0]
            non_zero_gt_boxes = non_zero_gt_boxes[:num_gt_boxes]
            all_rois = np.vstack((self.proposals, non_zero_gt_boxes))
            overlaps = calculate_bb_overlap(all_rois, non_zero_gt_boxes)
            gt_assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            labels = self.gt_classes.get()[:num_gt_boxes]
            labels = labels[gt_assignment]
            (keep_inds, fg_rois_this_img) = self._sample_fg_bg(max_overlaps)
            labels = labels[keep_inds]
            labels[fg_rois_this_img:] = 0
            rois = all_rois[keep_inds]
            targets = compute_targets(non_zero_gt_boxes[gt_assignment[keep_inds]], rois)
            targets = (targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS)
            num_proposals = rois.shape[0]
            (bbox_targets, bbox_inside_weights) = self._get_bbox_regression_labels(targets, labels)
            labels_full = self._onehot_labels(labels.ravel())
            frcn_labels[:] = labels_full
            labels_mask = np.zeros((self.num_rois, self.num_classes))
            labels_mask[:num_proposals, :] = 1.0
            frcn_labels_mask[:] = np.ascontiguousarray(labels_mask.T)
            frcn_bbtargets[:] = np.ascontiguousarray(bbox_targets.T)
            frcn_bbmask[:] = np.ascontiguousarray(bbox_inside_weights.T)
            rois = np.hstack([np.zeros((num_proposals, 1)), rois])
            rois = np.ascontiguousarray(rois, dtype=np.float32)
            self.dev_proposals_filtered.fill(0)
            self.dev_proposals_filtered[:num_proposals, :] = rois
            self.num_proposals = num_proposals
            return (inputs, self.dev_proposals_filtered.T)
        else:
            return (inputs, self.dev_proposals.T)

    def get_proposals(self):
        if False:
            return 10
        return (self.dev_proposals, self.num_proposals)

    def get_top_N_index(self, scores, N):
        if False:
            print('Hello World!')
        count = len(np.where(scores.get() > -1)[0])
        order = scores.get().ravel().argsort()[::-1].tolist()
        order = order[:count]
        if N > 0:
            order = order[:N]
        return order

    def bprop(self, errors, alpha=1.0, beta=0.0):
        if False:
            while True:
                i = 10
        'This layer propagate gradients from ROIs back to lower VGG layers'
        self.deltas = errors
        self.prev_layer.deltas[:] = errors
        return errors

    def _clip_boxes(self, boxes, im_shape):
        if False:
            return 10
        boxes[:, 0] = self.be.clip(boxes[:, 0], 0, im_shape[0] - 1)
        boxes[:, 1] = self.be.clip(boxes[:, 1], 0, im_shape[1] - 1)
        boxes[:, 2] = self.be.clip(boxes[:, 2], 0, im_shape[0] - 1)
        boxes[:, 3] = self.be.clip(boxes[:, 3], 0, im_shape[1] - 1)
        return boxes

    def _bbox_transform_inv(self, boxes, deltas, output):
        if False:
            i = 10
            return i + 15
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = self.be.exp(dw) * widths
        pred_h = self.be.exp(dh) * heights
        output[:, 0] = pred_ctr_x - 0.5 * pred_w
        output[:, 1] = pred_ctr_y - 0.5 * pred_h
        output[:, 2] = pred_ctr_x + 0.5 * pred_w
        output[:, 3] = pred_ctr_y + 0.5 * pred_h
        return output

    def _filter_boxes(self, boxes, min_size):
        if False:
            while True:
                i = 10
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = (ws >= min_size) * (hs >= min_size)
        return keep

    def _onehot_labels(self, labels):
        if False:
            i = 10
            return i + 15
        'Converts the roi labels from compressed (1, num_rois) shape\n        to the one-hot format required for the global buffers of shape\n        (num_classes, num_rois)'
        labels_full = np.zeros((self.num_classes, self.num_rois))
        for (idx, l) in enumerate(labels):
            labels_full[int(l), idx] = 1
        return labels_full

    def _get_bbox_regression_labels(self, bbox_target_data, labels):
        if False:
            for i in range(10):
                print('nop')
        'Bounding-box regression targets (bbox_target_data) are stored in a\n        compact form N x (tx, ty, tw, th)\n        This function expands those targets into the 4-of-4*K representation used\n        by the network (i.e. only one class has non-zero targets).\n        Returns:\n            bbox_targets (ndarray): N x 4K blob of regression targets\n            bbox_inside_weights (ndarray): N x 4K blob of loss weights\n        '
        bbox_targets = np.zeros((self.num_rois, 4 * self.num_classes), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(labels > 0)[0]
        for ind in inds:
            l = labels[ind]
            start = int(4 * l)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind]
            bbox_inside_weights[ind, start:end] = 1.0
        return (bbox_targets, bbox_inside_weights)

    def _sample_fg_bg(self, max_overlaps):
        if False:
            while True:
                i = 10
        'Return sample of at most fg_fraction * num_rois foreground indicies, padding\n        the remaining num_rois with background indicies. Foreground and background labels\n        are determined based on max_overlaps and the thresholds fg_thresh, bg_thresh_hi,\n        bg_thresh_lo.\n        Returns:\n            keep_inds (array): (num_rois,) sampled indicies of bboxes.\n            fg_rois_per_this_image (int): number of fg rois sampled from the image.\n        '
        fg_inds = np.where(max_overlaps >= self.fg_thresh)[0]
        fg_rois_per_image = np.round(self.fg_fraction * self.num_rois)
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        if fg_inds.size > 0 and (not self.deterministic):
            fg_inds = self.be.rng.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
        elif fg_inds.size > 0:
            fg_inds = fg_inds[:fg_rois_per_this_image]
        bg_inds = np.where((max_overlaps < self.bg_thresh_hi) & (max_overlaps >= self.bg_thresh_lo))[0]
        bg_rois_per_this_image = self.num_rois - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        if bg_inds.size > 0 and (not self.deterministic):
            bg_inds = self.be.rng.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)
        elif bg_inds.size > 0:
            bg_inds = bg_inds[:bg_rois_per_this_image]
        keep_inds = np.append(fg_inds, bg_inds)
        return (keep_inds, int(fg_rois_per_this_image))
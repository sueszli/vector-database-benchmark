import numpy as np
import torch
from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.utils.events import get_event_storage
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    if False:
        print('Hello World!')
    result_per_image = [fast_rcnn_inference_single_image(boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image) for (scores_per_image, boxes_per_image, image_shape) in zip(scores, boxes, image_shapes)]
    return tuple((list(x) for x in zip(*result_per_image)))

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):\n                box2box transform instance for proposal-to-detection transformations.\n            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class\n                logits for all R predicted object instances.\n                Each row corresponds to a predicted object instance.\n            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for\n                class-specific or class-agnostic regression. It stores the predicted deltas that\n                transform proposals into final box detections.\n                B is the box dimension (4 or 5).\n                When B is 4, each row is [dx, dy, dw, dh (, ....)].\n                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].\n            proposals (list[Instances]): A list of N Instances, where Instances i stores the\n                proposals for image i, in the field "proposal_boxes".\n                When training, each Instances must have ground-truth labels\n                stored in the field "gt_classes" and "gt_boxes".\n            smooth_l1_beta (float): The transition point between L1 and L2 loss in\n                the smooth L1 loss function. When set to 0, the loss becomes L1. When\n                set to +inf, the loss becomes constant 0.\n        '
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        box_type = type(proposals[0].proposal_boxes)
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, 'Proposals should not require gradients!'
        self.image_shapes = [x.image_size for x in proposals]
        if proposals[0].has('gt_boxes'):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has('gt_classes')
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        if False:
            return 10
        '\n        Log the accuracy metrics to EventStorage.\n        '
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]
        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
        storage = get_event_storage()
        storage.put_scalar('fast_rcnn/cls_accuracy', num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar('fast_rcnn/fg_cls_accuracy', fg_num_accurate / num_fg)
            storage.put_scalar('fast_rcnn/false_negative', num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the softmax cross entropy loss for box classification.\n\n        Returns:\n            scalar Tensor\n        '
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction='mean')

    def smooth_l1_loss(self):
        if False:
            print('Hello World!')
        '\n        Compute the smooth L1 loss for box regression.\n\n        Returns:\n            scalar Tensor\n        '
        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(1)
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
        loss_box_reg = smooth_l1_loss(self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols], gt_proposal_deltas[fg_inds], self.smooth_l1_beta, reduction='sum')
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the default losses for box head in Fast(er) R-CNN,\n        with softmax cross entropy loss and smooth L1 loss.\n\n        Returns:\n            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".\n        '
        return {'loss_cls': self.softmax_cross_entropy_loss(), 'loss_box_reg': self.smooth_l1_loss()}

    def predict_boxes(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes\n                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is\n                the number of predicted objects for image i and B is the box dimension (4 or 5)\n        '
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(self.pred_proposal_deltas.view(num_pred * K, B), self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B))
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        if False:
            print('Hello World!')
        '\n        Returns:\n            list[Tensor]: A list of Tensors of predicted class probabilities for each image.\n                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects\n                for image i.\n        '
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        if False:
            while True:
                i = 10
        '\n        Args:\n            score_thresh (float): same as fast_rcnn_inference.\n            nms_thresh (float): same as fast_rcnn_inference.\n            topk_per_image (int): same as fast_rcnn_inference.\n        Returns:\n            list[Instances]: same as fast_rcnn_inference.\n            list[Tensor]: same as fast_rcnn_inference.\n        '
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image)

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            cfg: config\n            input_size (int): channels, or (channels, height, width)\n            num_classes (int): number of foreground classes\n            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression\n            box_dim (int): the dimension of bounding boxes.\n                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes\n        '
        super(FastRCNNOutputLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for b in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(b.bias, 0)
        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        if False:
            while True:
                i = 10
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        if self._do_cls_dropout:
            x = F.dropout(x, self._dropout_ratio, training=self.training)
        scores = self.cls_score(x)
        return (scores, proposal_deltas)
"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/core/bbox/assigners
"""
import torch
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from modelscope.models.cv.object_detection_3d.depe.mmdet3d_plugin.core.bbox.util import normalize_bbox
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, cls_cost=dict(type='ClassificationCost', weight=1.0), reg_cost=dict(type='BBoxL1Cost', weight=1.0), iou_cost=dict(type='IoUCost', weight=0.0), pc_range=None):
        if False:
            print('Hello World!')
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-07):
        if False:
            print('Hello World!')
        "Computes one-to-one matching based on the weighted costs.\n        This method assign each query prediction to a ground truth or\n        background. The `assigned_gt_inds` with -1 means don't care,\n        0 means negative sample, and positive number is the index (1-based)\n        of assigned gt.\n        The assignment is done in the following steps, the order matters.\n        1. assign every prediction to -1\n        2. compute the weighted costs\n        3. do Hungarian matching on CPU based on the costs\n        4. assign all to 0 (background) first, then for each matched pair\n           between predictions and gts, treat this prediction as foreground\n           and assign the corresponding gt index (plus 1) to it.\n        Args:\n            bbox_pred (Tensor): Predicted boxes with normalized coordinates\n                (cx, cy, w, h), which are all in range [0, 1]. Shape\n                [num_query, 4].\n            cls_pred (Tensor): Predicted classification logits, shape\n                [num_query, num_class].\n            gt_bboxes (Tensor): Ground truth boxes with unnormalized\n                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].\n            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).\n            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are\n                labelled as `ignored`. Default None.\n            eps (int | float, optional): A value added to the denominator for\n                numerical stability. Default 1e-7.\n        Returns:\n            :obj:`AssignResult`: The assigned result.\n        "
        assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
        (num_gts, num_bboxes) = (gt_bboxes.size(0), bbox_pred.size(0))
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        cost = cls_cost + reg_cost
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        (matched_row_inds, matched_col_inds) = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
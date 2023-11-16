import torch
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector

@DETECTORS.register_module()
class GroupFree3DNet(SingleStage3DDetector):
    """`Group-Free 3D <https://arxiv.org/abs/2104.00678>`_."""

    def __init__(self, backbone, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        if False:
            i = 10
            return i + 15
        super(GroupFree3DNet, self).__init__(backbone=backbone, bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained)

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask=None, pts_instance_mask=None, gt_bboxes_ignore=None):
        if False:
            for i in range(10):
                print('nop')
        'Forward of training.\n\n        Args:\n            points (list[torch.Tensor]): Points of each batch.\n            img_metas (list): Image metas.\n            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.\n            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.\n            pts_semantic_mask (list[torch.Tensor]): point-wise semantic\n                label of each batch.\n            pts_instance_mask (list[torch.Tensor]): point-wise instance\n                label of each batch.\n            gt_bboxes_ignore (list[torch.Tensor]): Specify\n                which bounding.\n\n        Returns:\n            dict[str: torch.Tensor]: Losses.\n        '
        points_cat = torch.stack(points)
        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        if False:
            return 10
        'Forward of testing.\n\n        Args:\n            points (list[torch.Tensor]): Points of each sample.\n            img_metas (list): Image metas.\n            rescale (bool): Whether to rescale results.\n        Returns:\n            list: Predicted 3d boxes.\n        '
        points_cat = torch.stack(points)
        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for (bboxes, scores, labels) in bbox_list]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        if False:
            return 10
        'Test with augmentation.'
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)
        aug_bboxes = []
        for (x, pts_cat, img_meta) in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for (bboxes, scores, labels) in bbox_list]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.bbox_head.test_cfg)
        return [merged_bboxes]
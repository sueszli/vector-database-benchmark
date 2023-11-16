import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmdet.models.builder import HEADS
from ..utils import ConvModule_Norm
from .anchor_head import AnchorNHead

@HEADS.register_module()
class RPNNHead(AnchorNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """

    def __init__(self, in_channels, init_cfg=dict(type='Normal', layer='Conv2d', std=0.01), num_convs=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.num_convs = num_convs
        super(RPNNHead, self).__init__(1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        if False:
            i = 10
            return i + 15
        'Initialize layers of the head.'
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                rpn_convs.append(ConvModule_Norm(in_channels, self.feat_channels, 3, padding=1, norm_cfg=self.norm_cfg, inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def forward_single(self, x):
        if False:
            i = 10
            return i + 15
        'Forward feature map of a single scale level.'
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return (rpn_cls_score, rpn_bbox_pred)

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute losses of the head.\n\n        Args:\n            cls_scores (list[Tensor]): Box scores for each scale level\n                Has shape (N, num_anchors * num_classes, H, W)\n            bbox_preds (list[Tensor]): Box energies / deltas for each scale\n                level with shape (N, num_anchors * 4, H, W)\n            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with\n                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.\n            img_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n            gt_bboxes_ignore (None | list[Tensor]): specify which bounding\n                boxes can be ignored when computing the loss.\n\n        Returns:\n            dict[str, Tensor]: A dictionary of loss components.\n        '
        losses = super(RPNNHead, self).loss(cls_scores, bbox_preds, gt_bboxes, None, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_anchors, img_meta, cfg, rescale=False, with_nms=True, **kwargs):
        if False:
            i = 10
            return i + 15
        'Transform outputs of a single image into bbox predictions.\n\n        Args:\n            cls_score_list (list[Tensor]): Box scores from all scale\n                levels of a single image, each item has shape\n                (num_anchors * num_classes, H, W).\n            bbox_pred_list (list[Tensor]): Box energies / deltas from\n                all scale levels of a single image, each item has\n                shape (num_anchors * 4, H, W).\n            score_factor_list (list[Tensor]): Score factor from all scale\n                levels of a single image. RPN head does not need this value.\n            mlvl_anchors (list[Tensor]): Anchors of all scale level\n                each item has shape (num_anchors, 4).\n            img_meta (dict): Image meta info.\n            cfg (mmcv.Config): Test / postprocessing configuration,\n                if None, test_cfg would be used.\n            rescale (bool): If True, return boxes in original image space.\n                Default: False.\n            with_nms (bool): If True, do nms before return boxes.\n                Default: True.\n\n        Returns:\n            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns\n                are bounding box positions (tl_x, tl_y, br_x, br_y) and the\n                5-th column is a score between 0 and 1.\n        '
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                (ranked_scores, rank_inds) = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))
        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds, mlvl_valid_anchors, level_ids, cfg, img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors, level_ids, cfg, img_shape, **kwargs):
        if False:
            i = 10
            return i + 15
        'bbox post-processing method.\n\n        The boxes would be rescaled to the original image scale and do\n        the nms operation. Usually with_nms is False is used for aug test.\n\n        Args:\n            mlvl_scores (list[Tensor]): Box scores from all scale\n                levels of a single image, each item has shape\n                (num_bboxes, num_class).\n            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale\n                levels of a single image, each item has shape (num_bboxes, 4).\n            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level\n                each item has shape (num_bboxes, 4).\n            level_ids (list[Tensor]): Indexes from all scale levels of a\n                single image, each item has shape (num_bboxes, ).\n            cfg (mmcv.Config): Test / postprocessing configuration,\n                if None, test_cfg would be used.\n            img_shape (tuple(int)): Shape of current image.\n\n        Returns:\n            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns\n                are bounding box positions (tl_x, tl_y, br_x, br_y) and the\n                5-th column is a score between 0 and 1.\n        '
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            (dets, _) = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)
        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        if False:
            for i in range(10):
                print('nop')
        'Test without augmentation.\n\n        Args:\n            x (tuple[Tensor]): Features from the upstream network, each is\n                a 4D-tensor.\n            img_metas (list[dict]): Meta info of each image.\n        Returns:\n            Tensor: dets of shape [N, num_det, 5].\n        '
        (cls_scores, bbox_preds) = self(x)
        assert len(cls_scores) == len(bbox_preds)
        (batch_bboxes, batch_scores) = super(RPNNHead, self).onnx_export(cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        (dets, _) = add_dummy_nms_for_onnx(batch_bboxes, batch_scores, cfg.max_per_img, cfg.nms.iou_threshold, score_threshold, nms_pre, cfg.max_per_img)
        return dets
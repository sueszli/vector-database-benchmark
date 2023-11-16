import torch
from mmcv.cnn import Scale, bias_init_with_prob, normal_init
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn
from mmdet3d.core import build_prior_generator
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet.core import multi_apply, reduce_mean
from ..builder import HEADS, build_loss

@HEADS.register_module()
class ImVoxelHead(BaseModule):
    """`ImVoxelNet<https://arxiv.org/abs/2106.01178>`_ head for indoor
    datasets.

    Args:
        n_classes (int): Number of classes.
        n_levels (int): Number of feature levels.
        n_channels (int): Number of channels in input tensors.
        n_reg_outs (int): Number of regression layer channels.
        pts_assign_threshold (int): Min number of location per box to
            be assigned with.
        pts_center_threshold (int): Max number of locations per box to
            be assigned with.
        center_loss (dict, optional): Config of centerness loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict, optional): Config of bbox loss.
            Default: dict(type='RotatedIoU3DLoss').
        cls_loss (dict, optional): Config of classification loss.
            Default: dict(type='FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self, n_classes, n_levels, n_channels, n_reg_outs, pts_assign_threshold, pts_center_threshold, prior_generator, center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True), bbox_loss=dict(type='RotatedIoU3DLoss'), cls_loss=dict(type='FocalLoss'), train_cfg=None, test_cfg=None, init_cfg=None):
        if False:
            while True:
                i = 10
        super(ImVoxelHead, self).__init__(init_cfg)
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.prior_generator = build_prior_generator(prior_generator)
        self.center_loss = build_loss(center_loss)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        if False:
            while True:
                i = 10
        'Initialize neural network layers of the head.'
        self.conv_center = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)
        self.conv_reg = nn.Conv3d(n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.conv_cls = nn.Conv3d(n_channels, n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(n_levels)])

    def init_weights(self):
        if False:
            i = 10
            return i + 15
        'Initialize all layer weights.'
        normal_init(self.conv_center, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_init_with_prob(0.01))

    def _forward_single(self, x, scale):
        if False:
            i = 10
            return i + 15
        'Forward pass per level.\n\n        Args:\n            x (Tensor): Per level 3d neck output tensor.\n            scale (mmcv.cnn.Scale): Per level multiplication weight.\n\n        Returns:\n            tuple[Tensor]: Centerness, bbox and classification predictions.\n        '
        reg_final = self.conv_reg(x)
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        return (self.conv_center(x), bbox_pred, self.conv_cls(x))

    def forward(self, x):
        if False:
            return 10
        'Forward function.\n\n        Args:\n            x (list[Tensor]): Features from 3d neck.\n\n        Returns:\n            tuple[Tensor]: Centerness, bbox and classification predictions.\n        '
        return multi_apply(self._forward_single, x, self.scales)

    def _loss_single(self, center_preds, bbox_preds, cls_preds, valid_preds, img_meta, gt_bboxes, gt_labels):
        if False:
            for i in range(10):
                print('nop')
        'Per scene loss function.\n\n        Args:\n            center_preds (list[Tensor]): Centerness predictions for all levels.\n            bbox_preds (list[Tensor]): Bbox predictions for all levels.\n            cls_preds (list[Tensor]): Classification predictions for all\n                levels.\n            valid_preds (list[Tensor]): Valid mask predictions for all levels.\n            img_meta (dict): Scene meta info.\n            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.\n            gt_labels (Tensor): Ground truth labels.\n\n        Returns:\n            tuple[Tensor]: Centerness, bbox, and classification loss values.\n        '
        points = self._get_points(center_preds)
        (center_targets, bbox_targets, cls_targets) = self._get_targets(points, gt_bboxes, gt_labels)
        center_preds = torch.cat([x.permute(1, 2, 3, 0).reshape(-1) for x in center_preds])
        bbox_preds = torch.cat([x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in bbox_preds])
        cls_preds = torch.cat([x.permute(1, 2, 3, 0).reshape(-1, x.shape[0]) for x in cls_preds])
        valid_preds = torch.cat([x.permute(1, 2, 3, 0).reshape(-1) for x in valid_preds])
        points = torch.cat(points)
        pos_inds = torch.nonzero(torch.logical_and(cls_targets >= 0, valid_preds)).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.0)
        if torch.any(valid_preds):
            cls_loss = self.cls_loss(cls_preds[valid_preds], cls_targets[valid_preds], avg_factor=n_pos)
        else:
            cls_loss = cls_preds[valid_preds].sum()
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        if len(pos_inds) > 0:
            pos_center_targets = center_targets[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_points = points[pos_inds]
            center_loss = self.center_loss(pos_center_preds, pos_center_targets, avg_factor=n_pos)
            bbox_loss = self.bbox_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds), pos_bbox_targets, weight=pos_center_targets, avg_factor=pos_center_targets.sum())
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        return (center_loss, bbox_loss, cls_loss)

    def loss(self, center_preds, bbox_preds, cls_preds, valid_pred, gt_bboxes, gt_labels, img_metas):
        if False:
            for i in range(10):
                print('nop')
        'Per scene loss function.\n\n        Args:\n            center_preds (list[list[Tensor]]): Centerness predictions for\n                all scenes.\n            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.\n            cls_preds (list[list[Tensor]]): Classification predictions for all\n                scenes.\n            valid_pred (Tensor): Valid mask prediction for all scenes.\n            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth boxes for all\n                scenes.\n            gt_labels (list[Tensor]): Ground truth labels for all scenes.\n            img_metas (list[dict]): Meta infos for all scenes.\n\n        Returns:\n            dict: Centerness, bbox, and classification loss values.\n        '
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        (center_losses, bbox_losses, cls_losses) = ([], [], [])
        for i in range(len(img_metas)):
            (center_loss, bbox_loss, cls_loss) = self._loss_single(center_preds=[x[i] for x in center_preds], bbox_preds=[x[i] for x in bbox_preds], cls_preds=[x[i] for x in cls_preds], valid_preds=[x[i] for x in valid_preds], img_meta=img_metas[i], gt_bboxes=gt_bboxes[i], gt_labels=gt_labels[i])
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(center_loss=torch.mean(torch.stack(center_losses)), bbox_loss=torch.mean(torch.stack(bbox_losses)), cls_loss=torch.mean(torch.stack(cls_losses)))

    def _get_bboxes_single(self, center_preds, bbox_preds, cls_preds, valid_preds, img_meta):
        if False:
            return 10
        'Generate boxes for a single scene.\n\n        Args:\n            center_preds (list[Tensor]): Centerness predictions for all levels.\n            bbox_preds (list[Tensor]): Bbox predictions for all levels.\n            cls_preds (list[Tensor]): Classification predictions for all\n                levels.\n            valid_preds (list[Tensor]): Valid mask predictions for all levels.\n            img_meta (dict): Scene meta info.\n\n        Returns:\n            tuple[Tensor]: Predicted bounding boxes, scores and labels.\n        '
        points = self._get_points(center_preds)
        (mlvl_bboxes, mlvl_scores) = ([], [])
        for (center_pred, bbox_pred, cls_pred, valid_pred, point) in zip(center_preds, bbox_preds, cls_preds, valid_preds, points):
            center_pred = center_pred.permute(1, 2, 3, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(-1, bbox_pred.shape[0])
            cls_pred = cls_pred.permute(1, 2, 3, 0).reshape(-1, cls_pred.shape[0])
            valid_pred = valid_pred.permute(1, 2, 3, 0).reshape(-1, 1)
            scores = cls_pred.sigmoid() * center_pred.sigmoid() * valid_pred
            (max_scores, _) = scores.max(dim=1)
            if len(scores) > self.test_cfg.nms_pre > 0:
                (_, ids) = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]
            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        (bboxes, scores, labels) = self._single_scene_multiclass_nms(bboxes, scores, img_meta)
        return (bboxes, scores, labels)

    def get_bboxes(self, center_preds, bbox_preds, cls_preds, valid_pred, img_metas):
        if False:
            while True:
                i = 10
        'Generate boxes for all scenes.\n\n        Args:\n            center_preds (list[list[Tensor]]): Centerness predictions for\n                all scenes.\n            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.\n            cls_preds (list[list[Tensor]]): Classification predictions for all\n                scenes.\n            valid_pred (Tensor): Valid mask prediction for all scenes.\n            img_metas (list[dict]): Meta infos for all scenes.\n\n        Returns:\n            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for\n                all scenes.\n        '
        valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        results = []
        for i in range(len(img_metas)):
            results.append(self._get_bboxes_single(center_preds=[x[i] for x in center_preds], bbox_preds=[x[i] for x in bbox_preds], cls_preds=[x[i] for x in cls_preds], valid_preds=[x[i] for x in valid_preds], img_meta=img_metas[i]))
        return results

    @staticmethod
    def _upsample_valid_preds(valid_pred, features):
        if False:
            while True:
                i = 10
        'Upsample valid mask predictions.\n\n        Args:\n            valid_pred (Tensor): Valid mask prediction.\n            features (Tensor): Feature tensor.\n\n        Returns:\n            tuple[Tensor]: Upsampled valid masks for all feature levels.\n        '
        return [nn.Upsample(size=x.shape[-3:], mode='trilinear')(valid_pred).round().bool() for x in features]

    def _get_points(self, features):
        if False:
            for i in range(10):
                print('nop')
        'Generate final locations.\n\n        Args:\n            features (list[Tensor]): Feature tensors for all feature levels.\n\n        Returns:\n            list(Tensor): Final locations for all feature levels.\n        '
        points = []
        for x in features:
            n_voxels = x.size()[-3:][::-1]
            points.append(self.prior_generator.grid_anchors([n_voxels], device=x.device)[0][:, :3].reshape(n_voxels + (3,)).permute(2, 1, 0, 3).reshape(-1, 3))
        return points

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        if False:
            while True:
                i = 10
        'Transform predicted bbox parameters to bbox.\n\n        Args:\n            points (Tensor): Final locations of shape (N, 3).\n            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 7).\n\n        Returns:\n            Tensor: Transformed 3D box of shape (N, 7).\n        '
        if bbox_pred.shape[0] == 0:
            return bbox_pred
        shift = torch.stack(((bbox_pred[:, 1] - bbox_pred[:, 0]) / 2, (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2, (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2), dim=-1).view(-1, 1, 3)
        shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:, 0, :]
        center = points + shift
        size = torch.stack((bbox_pred[:, 0] + bbox_pred[:, 1], bbox_pred[:, 2] + bbox_pred[:, 3], bbox_pred[:, 4] + bbox_pred[:, 5]), dim=-1)
        return torch.cat((center, size, bbox_pred[:, 6:7]), dim=-1)

    @staticmethod
    def _get_face_distances(points, boxes):
        if False:
            i = 10
            return i + 15
        'Calculate distances from point to box faces.\n\n        Args:\n            points (Tensor): Final locations of shape (N_points, N_boxes, 3).\n            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)\n\n        Returns:\n            Tensor: Face distances of shape (N_points, N_boxes, 6),\n                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).\n        '
        shift = torch.stack((points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1], points[..., 2] - boxes[..., 2]), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        if False:
            while True:
                i = 10
        'Compute point centerness w.r.t containing box.\n\n        Args:\n            face_distances (Tensor): Face distances of shape (B, N, 6),\n                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).\n\n        Returns:\n            Tensor: Centerness of shape (B, N).\n        '
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    @torch.no_grad()
    def _get_targets(self, points, gt_bboxes, gt_labels):
        if False:
            i = 10
            return i + 15
        'Compute targets for final locations for a single scene.\n\n        Args:\n            points (list[Tensor]): Final locations for all levels.\n            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.\n            gt_labels (Tensor): Ground truth labels.\n\n        Returns:\n            tuple[Tensor]: Centerness, bbox and classification\n                targets for all locations.\n        '
        float_max = points[0].new_tensor(100000000.0)
        n_levels = len(points)
        levels = torch.cat([points[i].new_tensor(i).expand(len(points[i])) for i in range(len(points))])
        points = torch.cat(points)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(torch.sum(inside_box_condition[levels == i], dim=0))
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask), dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1, lower_index)
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels
        centerness = self._get_centerness(face_distances)
        centerness = torch.where(inside_box_condition, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(level_condition, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, min(self.pts_center_threshold + 1, len(centerness)), dim=0).values[-1]
        topk_condition = centerness > top_centerness.unsqueeze(0)
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        (min_volumes, min_inds) = volumes.min(dim=1)
        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        return (center_targets, bbox_targets, cls_targets)

    def _single_scene_multiclass_nms(self, bboxes, scores, input_meta):
        if False:
            i = 10
            return i + 15
        'Multi-class nms for a single scene.\n\n        Args:\n            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or\n                (N_boxes, 7).\n            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).\n            input_meta (dict): Scene meta data.\n\n        Returns:\n            tuple[Tensor]: Predicted bboxes, scores and labels.\n        '
        n_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        (nms_bboxes, nms_scores, nms_labels) = ([], [], [])
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue
            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms3d_normal
            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))
        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))
        if with_yaw:
            box_dim = 7
        else:
            box_dim = 6
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = input_meta['box_type_3d'](nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(0.5, 0.5, 0.5))
        return (nms_bboxes, nms_scores, nms_labels)
import torch
from mmcv.cnn import xavier_init
from torch import nn as nn
from mmdet3d.core.utils import get_ellip_gaussian_2D
from mmdet3d.models.model_utils import EdgeFusionModule
from mmdet3d.models.utils import filter_outside_objs, get_edge_indices, get_keypoints, handle_proj_objs
from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
from ..builder import HEADS, build_loss
from .anchor_free_mono3d_head import AnchorFreeMono3DHead

@HEADS.register_module()
class MonoFlexHead(AnchorFreeMono3DHead):
    """MonoFlex head used in `MonoFlex <https://arxiv.org/abs/2104.02323>`_

    .. code-block:: none

                / --> 3 x 3 conv --> 1 x 1 conv --> [edge fusion] --> cls
                |
                | --> 3 x 3 conv --> 1 x 1 conv --> 2d bbox
                |
                | --> 3 x 3 conv --> 1 x 1 conv --> [edge fusion] --> 2d offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints uncertainty
        feature
                | --> 3 x 3 conv --> 1 x 1 conv -->  keypoints uncertainty
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->   3d dimensions
                |
                |                  |--- 1 x 1 conv -->  ori cls
                | --> 3 x 3 conv --|
                |                  |--- 1 x 1 conv -->  ori offsets
                |
                | --> 3 x 3 conv --> 1 x 1 conv -->  depth
                |
                \\ --> 3 x 3 conv --> 1 x 1 conv -->  depth uncertainty

    Args:
        use_edge_fusion (bool): Whether to use edge fusion module while
            feature extraction.
        edge_fusion_inds (list[tuple]): Indices of feature to use edge fusion.
        edge_heatmap_ratio (float): Ratio of generating target heatmap.
        filter_outside_objs (bool, optional): Whether to filter the
            outside objects. Default: True.
        loss_cls (dict, optional): Config of classification loss.
            Default: loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0).
        loss_bbox (dict, optional): Config of localization loss.
            Default: loss_bbox=dict(type='IOULoss', loss_weight=10.0).
        loss_dir (dict, optional): Config of direction classification loss.
            Default: dict(type='MultibinLoss', loss_weight=0.1).
        loss_keypoints (dict, optional): Config of keypoints loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_dims: (dict, optional): Config of dimensions loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_offsets2d: (dict, optional): Config of offsets2d loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_direct_depth: (dict, optional): Config of directly regression depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_keypoints_depth: (dict, optional): Config of keypoints decoded depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_combined_depth: (dict, optional): Config of combined depth loss.
            Default: dict(type='L1Loss', loss_weight=0.1).
        loss_attr (dict, optional): Config of attribute classification loss.
            In MonoFlex, Default: None.
        bbox_coder (dict, optional): Bbox coder for encoding and decoding boxes.
            Default: dict(type='MonoFlexCoder', code_size=7).
        norm_cfg (dict, optional): Dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict): Initialization config dict. Default: None.
    """

    def __init__(self, num_classes, in_channels, use_edge_fusion, edge_fusion_inds, edge_heatmap_ratio, filter_outside_objs=True, loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0), loss_bbox=dict(type='IoULoss', loss_weight=0.1), loss_dir=dict(type='MultiBinLoss', loss_weight=0.1), loss_keypoints=dict(type='L1Loss', loss_weight=0.1), loss_dims=dict(type='L1Loss', loss_weight=0.1), loss_offsets2d=dict(type='L1Loss', loss_weight=0.1), loss_direct_depth=dict(type='L1Loss', loss_weight=0.1), loss_keypoints_depth=dict(type='L1Loss', loss_weight=0.1), loss_combined_depth=dict(type='L1Loss', loss_weight=0.1), loss_attr=None, bbox_coder=dict(type='MonoFlexCoder', code_size=7), norm_cfg=dict(type='BN'), init_cfg=None, init_bias=-2.19, **kwargs):
        if False:
            return 10
        self.use_edge_fusion = use_edge_fusion
        self.edge_fusion_inds = edge_fusion_inds
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dir=loss_dir, loss_attr=loss_attr, norm_cfg=norm_cfg, init_cfg=init_cfg, **kwargs)
        self.filter_outside_objs = filter_outside_objs
        self.edge_heatmap_ratio = edge_heatmap_ratio
        self.init_bias = init_bias
        self.loss_dir = build_loss(loss_dir)
        self.loss_keypoints = build_loss(loss_keypoints)
        self.loss_dims = build_loss(loss_dims)
        self.loss_offsets2d = build_loss(loss_offsets2d)
        self.loss_direct_depth = build_loss(loss_direct_depth)
        self.loss_keypoints_depth = build_loss(loss_keypoints_depth)
        self.loss_combined_depth = build_loss(loss_combined_depth)
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def _init_edge_module(self):
        if False:
            print('Hello World!')
        'Initialize edge fusion module for feature extraction.'
        self.edge_fuse_cls = EdgeFusionModule(self.num_classes, 256)
        for i in range(len(self.edge_fusion_inds)):
            (reg_inds, out_inds) = self.edge_fusion_inds[i]
            out_channels = self.group_reg_dims[reg_inds][out_inds]
            fusion_layer = EdgeFusionModule(out_channels, 256)
            layer_name = f'edge_fuse_reg_{reg_inds}_{out_inds}'
            self.add_module(layer_name, fusion_layer)

    def init_weights(self):
        if False:
            return 10
        'Initialize weights.'
        super().init_weights()
        self.conv_cls.bias.data.fill_(self.init_bias)
        xavier_init(self.conv_regs[4][0], gain=0.01)
        xavier_init(self.conv_regs[7][0], gain=0.01)
        for m in self.conv_regs.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_predictor(self):
        if False:
            i = 10
            return i + 15
        'Initialize predictor layers of the head.'
        self.conv_cls_prev = self._init_branch(conv_channels=self.cls_branch, conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dims = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            reg_list = nn.ModuleList()
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(self._init_branch(conv_channels=reg_branch_channels, conv_strides=(1,) * len(reg_branch_channels)))
                for reg_dim in reg_dims:
                    reg_list.append(nn.Conv2d(out_channel, reg_dim, 1))
                self.conv_regs.append(reg_list)
            else:
                self.conv_reg_prevs.append(None)
                for reg_dim in reg_dims:
                    reg_list.append(nn.Conv2d(self.feat_channels, reg_dim, 1))
                self.conv_regs.append(reg_list)

    def _init_layers(self):
        if False:
            print('Hello World!')
        'Initialize layers of the head.'
        self._init_predictor()
        if self.use_edge_fusion:
            self._init_edge_module()

    def forward_train(self, x, input_metas, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, attr_labels, gt_bboxes_ignore, proposal_cfg, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Args:\n            x (list[Tensor]): Features from FPN.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,\n                shape (num_gts, 4).\n            gt_labels (list[Tensor]): Ground truth labels of each box,\n                shape (num_gts,).\n            gt_bboxes_3d (list[Tensor]): 3D ground truth bboxes of the image,\n                shape (num_gts, self.bbox_code_size).\n            gt_labels_3d (list[Tensor]): 3D ground truth labels of each box,\n                shape (num_gts,).\n            centers2d (list[Tensor]): Projected 3D center of each box,\n                shape (num_gts, 2).\n            depths (list[Tensor]): Depth of projected 3D center of each box,\n                shape (num_gts,).\n            attr_labels (list[Tensor]): Attribute labels of each box,\n                shape (num_gts,).\n            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be\n                ignored, shape (num_ignored_gts, 4).\n            proposal_cfg (mmcv.Config): Test / postprocessing configuration,\n                if None, test_cfg would be used\n        Returns:\n            tuple:\n                losses: (dict[str, Tensor]): A dictionary of loss components.\n                proposal_list (list[Tensor]): Proposals of each image.\n        '
        outs = self(x, input_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_bboxes_3d, centers2d, depths, attr_labels, input_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, attr_labels, input_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, input_metas, cfg=proposal_cfg)
            return (losses, proposal_list)

    def forward(self, feats, input_metas):
        if False:
            return 10
        'Forward features from the upstream network.\n\n        Args:\n            feats (list[Tensor]): Features from the upstream network, each is\n                a 4D-tensor.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n\n        Returns:\n            tuple:\n                cls_scores (list[Tensor]): Box scores for each scale level,\n                    each is a 4D-tensor, the channel number is\n                    num_points * num_classes.\n                bbox_preds (list[Tensor]): Box energies / deltas for each scale\n                    level, each is a 4D-tensor, the channel number is\n                    num_points * bbox_code_size.\n        '
        mlvl_input_metas = [input_metas for i in range(len(feats))]
        return multi_apply(self.forward_single, feats, mlvl_input_metas)

    def forward_single(self, x, input_metas):
        if False:
            while True:
                i = 10
        'Forward features of a single scale level.\n\n        Args:\n            x (Tensor): Feature maps from a specific FPN feature level.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n\n        Returns:\n            tuple: Scores for each class, bbox predictions.\n        '
        (img_h, img_w) = input_metas[0]['pad_shape'][:2]
        (batch_size, _, feat_h, feat_w) = x.shape
        downsample_ratio = img_h / feat_h
        for conv_cls_prev_layer in self.conv_cls_prev:
            cls_feat = conv_cls_prev_layer(x)
        out_cls = self.conv_cls(cls_feat)
        if self.use_edge_fusion:
            edge_indices_list = get_edge_indices(input_metas, downsample_ratio, device=x.device)
            edge_lens = [edge_indices.shape[0] for edge_indices in edge_indices_list]
            max_edge_len = max(edge_lens)
            edge_indices = x.new_zeros((batch_size, max_edge_len, 2), dtype=torch.long)
            for i in range(batch_size):
                edge_indices[i, :edge_lens[i]] = edge_indices_list[i]
            out_cls = self.edge_fuse_cls(cls_feat, out_cls, edge_indices, edge_lens, feat_h, feat_w)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            reg_feat = x.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    reg_feat = conv_reg_prev_layer(reg_feat)
            for (j, conv_reg) in enumerate(self.conv_regs[i]):
                out_reg = conv_reg(reg_feat)
                if self.use_edge_fusion and (i, j) in self.edge_fusion_inds:
                    out_reg = getattr(self, 'edge_fuse_reg_{}_{}'.format(i, j))(reg_feat, out_reg, edge_indices, edge_lens, feat_h, feat_w)
                bbox_pred.append(out_reg)
        bbox_pred = torch.cat(bbox_pred, dim=1)
        cls_score = out_cls.sigmoid()
        cls_score = cls_score.clamp(min=0.0001, max=1 - 0.0001)
        return (cls_score, bbox_pred)

    def get_bboxes(self, cls_scores, bbox_preds, input_metas):
        if False:
            i = 10
            return i + 15
        'Generate bboxes from bbox head predictions.\n\n        Args:\n            cls_scores (list[Tensor]): Box scores for each scale level.\n            bbox_preds (list[Tensor]): Box regression for each scale.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n            rescale (bool): If True, return boxes in original image space.\n        Returns:\n            list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]:\n                Each item in result_list is 4-tuple.\n        '
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([cls_scores[0].new_tensor(input_meta['cam2img']) for input_meta in input_metas])
        (batch_bboxes, batch_scores, batch_topk_labels) = self.decode_heatmap(cls_scores[0], bbox_preds[0], input_metas, cam2imgs=cam2imgs, topk=100, kernel=3)
        result_list = []
        for img_id in range(len(input_metas)):
            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]
            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            bboxes = input_metas[img_id]['box_type_3d'](bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            attrs = None
            result_list.append((bboxes, scores, labels, attrs))
        return result_list

    def decode_heatmap(self, cls_score, reg_pred, input_metas, cam2imgs, topk=100, kernel=3):
        if False:
            return 10
        'Transform outputs into detections raw bbox predictions.\n\n        Args:\n            class_score (Tensor): Center predict heatmap,\n                shape (B, num_classes, H, W).\n            reg_pred (Tensor): Box regression map.\n                shape (B, channel, H , W).\n            input_metas (List[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n            cam2imgs (Tensor): Camera intrinsic matrix.\n                shape (N, 4, 4)\n            topk (int, optional): Get top k center keypoints from heatmap.\n                Default 100.\n            kernel (int, optional): Max pooling kernel for extract local\n                maximum pixels. Default 3.\n\n        Returns:\n            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing\n               the following Tensors:\n              - batch_bboxes (Tensor): Coords of each 3D box.\n                    shape (B, k, 7)\n              - batch_scores (Tensor): Scores of each 3D box.\n                    shape (B, k)\n              - batch_topk_labels (Tensor): Categories of each 3D box.\n                    shape (B, k)\n        '
        (img_h, img_w) = input_metas[0]['pad_shape'][:2]
        (batch_size, _, feat_h, feat_w) = cls_score.shape
        downsample_ratio = img_h / feat_h
        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)
        (*batch_dets, topk_ys, topk_xs) = get_topk_from_heatmap(center_heatmap_pred, k=topk)
        (batch_scores, batch_index, batch_topk_labels) = batch_dets
        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, 8)
        pred_base_centers2d = torch.cat([topk_xs.view(-1, 1), topk_ys.view(-1, 1).float()], dim=1)
        preds = self.bbox_coder.decode(regression, batch_topk_labels, downsample_ratio, cam2imgs)
        pred_locations = self.bbox_coder.decode_location(pred_base_centers2d, preds['offsets2d'], preds['combined_depth'], cam2imgs, downsample_ratio)
        pred_yaws = self.bbox_coder.decode_orientation(preds['orientations']).unsqueeze(-1)
        pred_dims = preds['dimensions']
        batch_bboxes = torch.cat((pred_locations, pred_dims, pred_yaws), dim=1)
        batch_bboxes = batch_bboxes.view(batch_size, -1, self.bbox_code_size)
        return (batch_bboxes, batch_scores, batch_topk_labels)

    def get_predictions(self, pred_reg, labels3d, centers2d, reg_mask, batch_indices, input_metas, downsample_ratio):
        if False:
            i = 10
            return i + 15
        'Prepare predictions for computing loss.\n\n        Args:\n            pred_reg (Tensor): Box regression map.\n                shape (B, channel, H , W).\n            labels3d (Tensor): Labels of each 3D box.\n                shape (B * max_objs, )\n            centers2d (Tensor): Coords of each projected 3D box\n                center on image. shape (N, 2)\n            reg_mask (Tensor): Indexes of the existence of the 3D box.\n                shape (B * max_objs, )\n            batch_indices (Tenosr): Batch indices of the 3D box.\n                shape (N, 3)\n            input_metas (list[dict]): Meta information of each image,\n                e.g., image size, scaling factor, etc.\n            downsample_ratio (int): The stride of feature map.\n\n        Returns:\n            dict: The predictions for computing loss.\n        '
        (batch, channel) = (pred_reg.shape[0], pred_reg.shape[1])
        w = pred_reg.shape[3]
        cam2imgs = torch.stack([centers2d.new_tensor(input_meta['cam2img']) for input_meta in input_metas])
        cam2imgs = cam2imgs[batch_indices, :, :]
        centers2d_inds = centers2d[:, 1] * w + centers2d[:, 0]
        centers2d_inds = centers2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)[reg_mask]
        preds = self.bbox_coder.decode(pred_regression_pois, labels3d, downsample_ratio, cam2imgs)
        return preds

    def get_targets(self, gt_bboxes_list, gt_labels_list, gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list, depths_list, feat_shape, img_shape, input_metas):
        if False:
            return 10
        'Get training targets for batch images.\n``\n        Args:\n            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each\n                image, shape (num_gt, 4).\n            gt_labels_list (list[Tensor]): Ground truth labels of each\n                box, shape (num_gt,).\n            gt_bboxes_3d_list (list[:obj:`CameraInstance3DBoxes`]): 3D\n                Ground truth bboxes of each image,\n                shape (num_gt, bbox_code_size).\n            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of\n                each box, shape (num_gt,).\n            centers2d_list (list[Tensor]): Projected 3D centers onto 2D\n                image, shape (num_gt, 2).\n            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D\n                image, each has shape (num_gt, 1).\n            feat_shape (tuple[int]): Feature map shape with value,\n                shape (B, _, H, W).\n            img_shape (tuple[int]): Image shape in [h, w] format.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n\n        Returns:\n            tuple[Tensor, dict]: The Tensor value is the targets of\n                center heatmap, the dict has components below:\n              - base_centers2d_target (Tensor): Coords of each projected 3D box\n                    center on image. shape (B * max_objs, 2), [dtype: int]\n              - labels3d (Tensor): Labels of each 3D box.\n                    shape (N, )\n              - reg_mask (Tensor): Mask of the existence of the 3D box.\n                    shape (B * max_objs, )\n              - batch_indices (Tensor): Batch id of the 3D box.\n                    shape (N, )\n              - depth_target (Tensor): Depth target of each 3D box.\n                    shape (N, )\n              - keypoints2d_target (Tensor): Keypoints of each projected 3D box\n                    on image. shape (N, 10, 2)\n              - keypoints_mask (Tensor): Keypoints mask of each projected 3D\n                    box on image. shape (N, 10)\n              - keypoints_depth_mask (Tensor): Depths decoded from keypoints\n                    of each 3D box. shape (N, 3)\n              - orientations_target (Tensor): Orientation (encoded local yaw)\n                    target of each 3D box. shape (N, )\n              - offsets2d_target (Tensor): Offsets target of each projected\n                    3D box. shape (N, 2)\n              - dimensions_target (Tensor): Dimensions target of each 3D box.\n                    shape (N, 3)\n              - downsample_ratio (int): The stride of feature map.\n        '
        (img_h, img_w) = img_shape[:2]
        (batch_size, _, feat_h, feat_w) = feat_shape
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        assert width_ratio == height_ratio
        if self.filter_outside_objs:
            filter_outside_objs(gt_bboxes_list, gt_labels_list, gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list, input_metas)
        (base_centers2d_list, offsets2d_list, trunc_mask_list) = handle_proj_objs(centers2d_list, gt_bboxes_list, input_metas)
        (keypoints2d_list, keypoints_mask_list, keypoints_depth_mask_list) = get_keypoints(gt_bboxes_3d_list, centers2d_list, input_metas)
        center_heatmap_target = gt_bboxes_list[-1].new_zeros([batch_size, self.num_classes, feat_h, feat_w])
        for batch_id in range(batch_size):
            gt_bboxes = gt_bboxes_list[batch_id] * width_ratio
            gt_labels = gt_labels_list[batch_id]
            gt_base_centers2d = base_centers2d_list[batch_id] * width_ratio
            trunc_masks = trunc_mask_list[batch_id]
            for (j, base_center2d) in enumerate(gt_base_centers2d):
                if trunc_masks[j]:
                    (base_center2d_x_int, base_center2d_y_int) = base_center2d.int()
                    scale_box_w = min(base_center2d_x_int - gt_bboxes[j][0], gt_bboxes[j][2] - base_center2d_x_int)
                    scale_box_h = min(base_center2d_y_int - gt_bboxes[j][1], gt_bboxes[j][3] - base_center2d_y_int)
                    radius_x = scale_box_w * self.edge_heatmap_ratio
                    radius_y = scale_box_h * self.edge_heatmap_ratio
                    (radius_x, radius_y) = (max(0, int(radius_x)), max(0, int(radius_y)))
                    assert min(radius_x, radius_y) == 0
                    ind = gt_labels[j]
                    get_ellip_gaussian_2D(center_heatmap_target[batch_id, ind], [base_center2d_x_int, base_center2d_y_int], radius_x, radius_y)
                else:
                    (base_center2d_x_int, base_center2d_y_int) = base_center2d.int()
                    scale_box_h = gt_bboxes[j][3] - gt_bboxes[j][1]
                    scale_box_w = gt_bboxes[j][2] - gt_bboxes[j][0]
                    radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.7)
                    radius = max(0, int(radius))
                    ind = gt_labels[j]
                    gen_gaussian_target(center_heatmap_target[batch_id, ind], [base_center2d_x_int, base_center2d_y_int], radius)
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        num_ctrs = [centers2d.shape[0] for centers2d in centers2d_list]
        max_objs = max(num_ctrs)
        batch_indices = [centers2d_list[0].new_full((num_ctrs[i],), i) for i in range(batch_size)]
        batch_indices = torch.cat(batch_indices, dim=0)
        reg_mask = torch.zeros((batch_size, max_objs), dtype=torch.bool).to(base_centers2d_list[0].device)
        gt_bboxes_3d = input_metas['box_type_3d'].cat(gt_bboxes_3d_list)
        gt_bboxes_3d = gt_bboxes_3d.to(base_centers2d_list[0].device)
        orienations_target = self.bbox_coder.encode(gt_bboxes_3d)
        batch_base_centers2d = base_centers2d_list[0].new_zeros((batch_size, max_objs, 2))
        for i in range(batch_size):
            reg_mask[i, :num_ctrs[i]] = 1
            batch_base_centers2d[i, :num_ctrs[i]] = base_centers2d_list[i]
        flatten_reg_mask = reg_mask.flatten()
        batch_base_centers2d = batch_base_centers2d.view(-1, 2) * width_ratio
        dimensions_target = gt_bboxes_3d.tensor[:, 3:6]
        labels_3d = torch.cat(gt_labels_3d_list)
        keypoints2d_target = torch.cat(keypoints2d_list)
        keypoints_mask = torch.cat(keypoints_mask_list)
        keypoints_depth_mask = torch.cat(keypoints_depth_mask_list)
        offsets2d_target = torch.cat(offsets2d_list)
        bboxes2d = torch.cat(gt_bboxes_list)
        bboxes2d_target = torch.cat([bboxes2d[:, 0:2] * -1, bboxes2d[:, 2:]], dim=-1)
        depths = torch.cat(depths_list)
        target_labels = dict(base_centers2d_target=batch_base_centers2d.int(), labels3d=labels_3d, reg_mask=flatten_reg_mask, batch_indices=batch_indices, bboxes2d_target=bboxes2d_target, depth_target=depths, keypoints2d_target=keypoints2d_target, keypoints_mask=keypoints_mask, keypoints_depth_mask=keypoints_depth_mask, orienations_target=orienations_target, offsets2d_target=offsets2d_target, dimensions_target=dimensions_target, downsample_ratio=1 / width_ratio)
        return (center_heatmap_target, avg_factor, target_labels)

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, attr_labels, input_metas, gt_bboxes_ignore=None):
        if False:
            while True:
                i = 10
        "Compute loss of the head.\n\n        Args:\n            cls_scores (list[Tensor]): Box scores for each scale level.\n                shape (num_gt, 4).\n            bbox_preds (list[Tensor]): Box dims is a 4D-tensor, the channel\n                number is bbox_code_size.\n                shape (B, 7, H, W).\n            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.\n                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.\n            gt_labels (list[Tensor]): Class indices corresponding to each box.\n                shape (num_gts, ).\n            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D boxes ground\n                truth. it is the flipped gt_bboxes\n            gt_labels_3d (list[Tensor]): Same as gt_labels.\n            centers2d (list[Tensor]): 2D centers on the image.\n                shape (num_gts, 2).\n            depths (list[Tensor]): Depth ground truth.\n                shape (num_gts, ).\n            attr_labels (list[Tensor]): Attributes indices of each box.\n                In kitti it's None.\n            input_metas (list[dict]): Meta information of each image, e.g.,\n                image size, scaling factor, etc.\n            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding\n                boxes can be ignored when computing the loss.\n                Default: None.\n\n        Returns:\n            dict[str, Tensor]: A dictionary of loss components.\n        "
        assert len(cls_scores) == len(bbox_preds) == 1
        assert attr_labels is None
        assert gt_bboxes_ignore is None
        center2d_heatmap = cls_scores[0]
        pred_reg = bbox_preds[0]
        (center2d_heatmap_target, avg_factor, target_labels) = self.get_targets(gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, center2d_heatmap.shape, input_metas[0]['pad_shape'], input_metas)
        preds = self.get_predictions(pred_reg=pred_reg, labels3d=target_labels['labels3d'], centers2d=target_labels['base_centers2d_target'], reg_mask=target_labels['reg_mask'], batch_indices=target_labels['batch_indices'], input_metas=input_metas, downsample_ratio=target_labels['downsample_ratio'])
        loss_cls = self.loss_cls(center2d_heatmap, center2d_heatmap_target, avg_factor=avg_factor)
        loss_bbox = self.loss_bbox(preds['bboxes2d'], target_labels['bboxes2d_target'])
        keypoints2d_mask = target_labels['keypoints2d_mask']
        loss_keypoints = self.loss_keypoints(preds['keypoints2d'][keypoints2d_mask], target_labels['keypoints2d_target'][keypoints2d_mask])
        loss_dir = self.loss_dir(preds['orientations'], target_labels['orientations_target'])
        loss_dims = self.loss_dims(preds['dimensions'], target_labels['dimensions_target'])
        loss_offsets2d = self.loss_offsets2d(preds['offsets2d'], target_labels['offsets2d_target'])
        direct_depth_weights = torch.exp(-preds['direct_depth_uncertainty'])
        loss_weight_1 = self.loss_direct_depth.loss_weight
        loss_direct_depth = self.loss_direct_depth(preds['direct_depth'], target_labels['depth_target'], direct_depth_weights)
        loss_uncertainty_1 = preds['direct_depth_uncertainty'] * loss_weight_1
        loss_direct_depth = loss_direct_depth + loss_uncertainty_1.mean()
        depth_mask = target_labels['keypoints_depth_mask']
        depth_target = target_labels['depth_target'].unsqueeze(-1).repeat(1, 3)
        valid_keypoints_depth_uncertainty = preds['keypoints_depth_uncertainty'][depth_mask]
        valid_keypoints_depth_weights = torch.exp(-valid_keypoints_depth_uncertainty)
        loss_keypoints_depth = self.loss_keypoint_depth(preds['keypoints_depth'][depth_mask], depth_target[depth_mask], valid_keypoints_depth_weights)
        loss_weight_2 = self.loss_keypoints_depth.loss_weight
        loss_uncertainty_2 = valid_keypoints_depth_uncertainty * loss_weight_2
        loss_keypoints_depth = loss_keypoints_depth + loss_uncertainty_2.mean()
        loss_combined_depth = self.loss_combined_depth(preds['combined_depth'], target_labels['depth_target'])
        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_keypoints=loss_keypoints, loss_dir=loss_dir, loss_dims=loss_dims, loss_offsets2d=loss_offsets2d, loss_direct_depth=loss_direct_depth, loss_keypoints_depth=loss_keypoints_depth, loss_combined_depth=loss_combined_depth)
        return loss_dict
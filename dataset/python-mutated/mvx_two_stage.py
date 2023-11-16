import warnings
from os import path as osp
import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F
from mmdet3d.core import Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result
from mmdet.core import multi_apply
from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector

@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, pts_voxel_layer=None, pts_voxel_encoder=None, pts_middle_encoder=None, pts_fusion_layer=None, img_backbone=None, pts_backbone=None, img_neck=None, pts_neck=None, pts_bbox_head=None, img_roi_head=None, img_rpn_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if False:
            return 10
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated key, please consider using init_cfg.')
                self.img_backbone.init_cfg = dict(type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated key, please consider using init_cfg.')
                self.img_roi_head.init_cfg = dict(type='Pretrained', checkpoint=img_pretrained)
        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(type='Pretrained', checkpoint=pts_pretrained)

    @property
    def with_img_shared_head(self):
        if False:
            while True:
                i = 10
        'bool: Whether the detector has a shared head in image branch.'
        return hasattr(self, 'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        if False:
            print('Hello World!')
        'bool: Whether the detector has a 3D box head.'
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a 2D image box head.'
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a 2D image backbone.'
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        if False:
            i = 10
            return i + 15
        'bool: Whether the detector has a 3D backbone.'
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a fusion layer.'
        return hasattr(self, 'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        if False:
            return 10
        'bool: Whether the detector has a neck in image branch.'
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a neck in 3D detector branch.'
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        if False:
            i = 10
            return i + 15
        'bool: Whether the detector has a 2D RPN in image detector branch.'
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        if False:
            return 10
        'bool: Whether the detector has a RoI Head in image branch.'
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        if False:
            i = 10
            return i + 15
        'bool: Whether the detector has a voxel encoder.'
        return hasattr(self, 'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a middle encoder.'
        return hasattr(self, 'middle_encoder') and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        if False:
            print('Hello World!')
        'Extract features of images.'
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                (B, N, C, H, W) = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        if False:
            for i in range(10):
                print('nop')
        'Extract features of points.'
        if not self.with_pts_bbox:
            return None
        (voxels, num_points, coors) = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        if False:
            return 10
        'Extract features from images and points.'
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        if False:
            return 10
        'Apply dynamic voxelization to points.\n\n        Args:\n            points (list[torch.Tensor]): Points of each sample.\n\n        Returns:\n            tuple[torch.Tensor]: Concatenated points, number of points\n                per voxel, and coordinates.\n        '
        (voxels, coors, num_points) = ([], [], [])
        for res in points:
            (res_voxels, res_coors, res_num_points) = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for (i, coor) in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return (voxels, num_points, coors_batch)

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None, gt_labels=None, gt_bboxes=None, img=None, proposals=None, gt_bboxes_ignore=None):
        if False:
            while True:
                i = 10
        'Forward training function.\n\n        Args:\n            points (list[torch.Tensor], optional): Points of each sample.\n                Defaults to None.\n            img_metas (list[dict], optional): Meta information of each sample.\n                Defaults to None.\n            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):\n                Ground truth 3D boxes. Defaults to None.\n            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels\n                of 3D boxes. Defaults to None.\n            gt_labels (list[torch.Tensor], optional): Ground truth labels\n                of 2D boxes in images. Defaults to None.\n            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in\n                images. Defaults to None.\n            img (torch.Tensor, optional): Images of each sample with shape\n                (N, C, H, W). Defaults to None.\n            proposals ([list[torch.Tensor], optional): Predicted proposals\n                used for training Fast RCNN. Defaults to None.\n            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth\n                2D boxes in images to be ignored. Defaults to None.\n\n        Returns:\n            dict: Losses of different branches.\n        '
        (img_feats, pts_feats) = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(img_feats, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore, proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None):
        if False:
            while True:
                i = 10
        'Forward function for point cloud branch.\n\n        Args:\n            pts_feats (list[torch.Tensor]): Features of point cloud branch\n            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth\n                boxes for each sample.\n            gt_labels_3d (list[torch.Tensor]): Ground truth labels for\n                boxes of each sampole\n            img_metas (list[dict]): Meta information of samples.\n            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth\n                boxes to be ignored. Defaults to None.\n\n        Returns:\n            dict: Losses of each branch.\n        '
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, proposals=None, **kwargs):
        if False:
            print('Hello World!')
        'Forward function for image branch.\n\n        This function works similar to the forward function of Faster R-CNN.\n\n        Args:\n            x (list[torch.Tensor]): Image features of shape (B, C, H, W)\n                of multiple levels.\n            img_metas (list[dict]): Meta information of images.\n            gt_bboxes (list[torch.Tensor]): Ground truth boxes of each image\n                sample.\n            gt_labels (list[torch.Tensor]): Ground truth labels of boxes.\n            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth\n                boxes to be ignored. Defaults to None.\n            proposals (list[torch.Tensor], optional): Proposals of each sample.\n                Defaults to None.\n\n        Returns:\n            dict: Losses of each branch.\n        '
        losses = dict()
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas, self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('img_rpn_proposal', self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if self.with_img_bbox:
            img_roi_losses = self.img_roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)
        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        if False:
            print('Hello World!')
        'Test without augmentation.'
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas, self.test_cfg.img_rpn)
        else:
            proposal_list = proposals
        return self.img_roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        if False:
            print('Hello World!')
        'RPN test function.'
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        if False:
            print('Hello World!')
        'Test function of point cloud branch.'
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for (bboxes, scores, labels) in bbox_list]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        if False:
            while True:
                i = 10
        'Test function without augmentaiton.'
        (img_feats, pts_feats) = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for (result_dict, pts_bbox) in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for (result_dict, img_bbox) in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        if False:
            return 10
        'Test function with augmentaiton.'
        (img_feats, pts_feats) = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        if False:
            print('Hello World!')
        'Extract point and image features of multiple samples.'
        if imgs is None:
            imgs = [None] * len(img_metas)
        (img_feats, pts_feats) = multi_apply(self.extract_feat, points, imgs, img_metas)
        return (img_feats, pts_feats)

    def aug_test_pts(self, feats, img_metas, rescale=False):
        if False:
            for i in range(10):
                print('nop')
        'Test function of point cloud branch with augmentaiton.'
        aug_bboxes = []
        for (x, img_meta) in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for (bboxes, scores, labels) in bbox_list]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir):
        if False:
            return 10
        'Results visualization.\n\n        Args:\n            data (dict): Input points and the information of the sample.\n            result (dict): Prediction results.\n            out_dir (str): Output directory of visualization result.\n        '
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} for visualization!")
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id]['box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(f"Unsupported data type {type(data['img_metas'][0])} for visualization!")
            file_name = osp.split(pts_filename)[-1].split('.')[0]
            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            if box_mode_3d == Box3DMode.CAM or box_mode_3d == Box3DMode.LIDAR:
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f'Unsupported box_mode_3d {box_mode_3d} for conversion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)
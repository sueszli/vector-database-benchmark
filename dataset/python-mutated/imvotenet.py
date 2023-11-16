import warnings
import numpy as np
import torch
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.utils import MLP
from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector

def sample_valid_seeds(mask, num_sampled_seed=1024):
    if False:
        while True:
            i = 10
    'Randomly sample seeds from all imvotes.\n\n    Modified from `<https://github.com/facebookresearch/imvotenet/blob/a8856345146bacf29a57266a2f0b874406fd8823/models/imvotenet.py#L26>`_\n\n    Args:\n        mask (torch.Tensor): Bool tensor in shape (\n            seed_num*max_imvote_per_pixel), indicates\n            whether this imvote corresponds to a 2D bbox.\n        num_sampled_seed (int): How many to sample from all imvotes.\n\n    Returns:\n        torch.Tensor: Indices with shape (num_sampled_seed).\n    '
    device = mask.device
    batch_size = mask.shape[0]
    sample_inds = mask.new_zeros((batch_size, num_sampled_seed), dtype=torch.int64)
    for bidx in range(batch_size):
        valid_inds = torch.nonzero(mask[bidx, :]).squeeze(-1)
        if len(valid_inds) < num_sampled_seed:
            t1 = torch.arange(num_sampled_seed, device=device)
            t2 = valid_inds % num_sampled_seed
            combined = torch.cat((t1, t2))
            (uniques, counts) = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            rand_inds = torch.randperm(len(difference), device=device)[:num_sampled_seed - len(valid_inds)]
            cur_sample_inds = difference[rand_inds]
            cur_sample_inds = torch.cat((valid_inds, cur_sample_inds))
        else:
            rand_inds = torch.randperm(len(valid_inds), device=device)[:num_sampled_seed]
            cur_sample_inds = valid_inds[rand_inds]
        sample_inds[bidx, :] = cur_sample_inds
    return sample_inds

@DETECTORS.register_module()
class ImVoteNet(Base3DDetector):
    """`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection."""

    def __init__(self, pts_backbone=None, pts_bbox_heads=None, pts_neck=None, img_backbone=None, img_neck=None, img_roi_head=None, img_rpn_head=None, img_mlp=None, freeze_img_branch=False, fusion_layer=None, num_sampled_seed=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if False:
            print('Hello World!')
        super(ImVoteNet, self).__init__(init_cfg=init_cfg)
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_heads is not None:
            pts_bbox_head_common = pts_bbox_heads.common
            pts_bbox_head_common.update(train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_common.update(test_cfg=test_cfg.pts)
            pts_bbox_head_joint = pts_bbox_head_common.copy()
            pts_bbox_head_joint.update(pts_bbox_heads.joint)
            pts_bbox_head_pts = pts_bbox_head_common.copy()
            pts_bbox_head_pts.update(pts_bbox_heads.pts)
            pts_bbox_head_img = pts_bbox_head_common.copy()
            pts_bbox_head_img.update(pts_bbox_heads.img)
            self.pts_bbox_head_joint = builder.build_head(pts_bbox_head_joint)
            self.pts_bbox_head_pts = builder.build_head(pts_bbox_head_pts)
            self.pts_bbox_head_img = builder.build_head(pts_bbox_head_img)
            self.pts_bbox_heads = [self.pts_bbox_head_joint, self.pts_bbox_head_pts, self.pts_bbox_head_img]
            self.loss_weights = pts_bbox_heads.loss_weights
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = builder.build_head(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg is not None else None
            img_roi_head.update(train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = builder.build_head(img_roi_head)
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
            self.max_imvote_per_pixel = fusion_layer.max_imvote_per_pixel
        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()
        if img_mlp is not None:
            self.img_mlp = MLP(**img_mlp)
        self.num_sampled_seed = num_sampled_seed
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
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated key, please consider using init_cfg.')
                self.pts_backbone.init_cfg = dict(type='Pretrained', checkpoint=pts_pretrained)

    def freeze_img_branch_params(self):
        if False:
            for i in range(10):
                print('nop')
        'Freeze all image branch parameters.'
        if self.with_img_bbox_head:
            for param in self.img_bbox_head.parameters():
                param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.with_img_rpn:
            for param in self.img_rpn_head.parameters():
                param.requires_grad = False
        if self.with_img_roi_head:
            for param in self.img_roi_head.parameters():
                param.requires_grad = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if False:
            print('Hello World!')
        'Overload in order to load img network ckpts into img branch.'
        module_names = ['backbone', 'neck', 'roi_head', 'rpn_head']
        for key in list(state_dict):
            for module_name in module_names:
                if key.startswith(module_name) and 'img_' + key not in state_dict:
                    state_dict['img_' + key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        if False:
            print('Hello World!')
        'Overload in order to keep image branch modules in eval mode.'
        super(ImVoteNet, self).train(mode)
        if self.freeze_img_branch:
            if self.with_img_bbox_head:
                self.img_bbox_head.eval()
            if self.with_img_backbone:
                self.img_backbone.eval()
            if self.with_img_neck:
                self.img_neck.eval()
            if self.with_img_rpn:
                self.img_rpn_head.eval()
            if self.with_img_roi_head:
                self.img_roi_head.eval()

    @property
    def with_img_bbox(self):
        if False:
            i = 10
            return i + 15
        'bool: Whether the detector has a 2D image box head.'
        return hasattr(self, 'img_roi_head') and self.img_roi_head.with_bbox or (hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None)

    @property
    def with_img_bbox_head(self):
        if False:
            while True:
                i = 10
        'bool: Whether the detector has a 2D image box head (not roi).'
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        if False:
            while True:
                i = 10
        'bool: Whether the detector has a 2D image backbone.'
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        if False:
            return 10
        'bool: Whether the detector has a neck in image branch.'
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_rpn(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a 2D RPN in image detector branch.'
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a RoI Head in image branch.'
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_bbox(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Whether the detector has a 3D box head.'
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_pts_backbone(self):
        if False:
            i = 10
            return i + 15
        'bool: Whether the detector has a 3D backbone.'
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        if False:
            return 10
        'bool: Whether the detector has a neck in 3D detector branch.'
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_feat(self, imgs):
        if False:
            return 10
        'Just to inherit from abstract method.'
        pass

    def extract_img_feat(self, img):
        if False:
            i = 10
            return i + 15
        'Directly extract features from the img backbone+neck.'
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_img_feats(self, imgs):
        if False:
            i = 10
            return i + 15
        'Extract features from multiple images.\n\n        Args:\n            imgs (list[torch.Tensor]): A list of images. The images are\n                augmented from the same image but in different ways.\n\n        Returns:\n            list[torch.Tensor]: Features of different images\n        '
        assert isinstance(imgs, list)
        return [self.extract_img_feat(img) for img in imgs]

    def extract_pts_feat(self, pts):
        if False:
            for i in range(10):
                print('nop')
        'Extract features of points.'
        x = self.pts_backbone(pts)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]
        return (seed_points, seed_features, seed_indices)

    def extract_pts_feats(self, pts):
        if False:
            return 10
        'Extract features of points from multiple samples.'
        assert isinstance(pts, list)
        return [self.extract_pts_feat(pt) for pt in pts]

    @torch.no_grad()
    def extract_bboxes_2d(self, img, img_metas, train=True, bboxes_2d=None, **kwargs):
        if False:
            while True:
                i = 10
        'Extract bounding boxes from 2d detector.\n\n        Args:\n            img (torch.Tensor): of shape (N, C, H, W) encoding input images.\n                Typically these should be mean centered and std scaled.\n            img_metas (list[dict]): Image meta info.\n            train (bool): train-time or not.\n            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,\n                not supported yet.\n\n        Return:\n            list[torch.Tensor]: a list of processed 2d bounding boxes.\n        '
        if bboxes_2d is None:
            x = self.extract_img_feat(img)
            proposal_list = self.img_rpn_head.simple_test_rpn(x, img_metas)
            rets = self.img_roi_head.simple_test(x, proposal_list, img_metas, rescale=False)
            rets_processed = []
            for ret in rets:
                tmp = np.concatenate(ret, axis=0)
                sem_class = img.new_zeros(len(tmp))
                start = 0
                for (i, bboxes) in enumerate(ret):
                    sem_class[start:start + len(bboxes)] = i
                    start += len(bboxes)
                ret = img.new_tensor(tmp)
                ret = torch.cat([ret, sem_class[:, None]], dim=-1)
                inds = torch.argsort(ret[:, 4], descending=True)
                ret = ret.index_select(0, inds)
                if train:
                    rand_drop = torch.randperm(len(ret))[:(len(ret) + 1) // 2]
                    rand_drop = torch.sort(rand_drop)[0]
                    ret = ret[rand_drop]
                rets_processed.append(ret.float())
            return rets_processed
        else:
            rets_processed = []
            for ret in bboxes_2d:
                if len(ret) > 0 and train:
                    rand_drop = torch.randperm(len(ret))[:(len(ret) + 1) // 2]
                    rand_drop = torch.sort(rand_drop)[0]
                    ret = ret[rand_drop]
                rets_processed.append(ret.float())
            return rets_processed

    def forward_train(self, points=None, img=None, img_metas=None, gt_bboxes=None, gt_labels=None, gt_bboxes_ignore=None, gt_masks=None, proposals=None, bboxes_2d=None, gt_bboxes_3d=None, gt_labels_3d=None, pts_semantic_mask=None, pts_instance_mask=None, **kwargs):
        if False:
            while True:
                i = 10
        "Forwarding of train for image branch pretrain or stage 2 train.\n\n        Args:\n            points (list[torch.Tensor]): Points of each batch.\n            img (torch.Tensor): of shape (N, C, H, W) encoding input images.\n                Typically these should be mean centered and std scaled.\n            img_metas (list[dict]): list of image and point cloud meta info\n                dict. For example, keys include 'ori_shape', 'img_norm_cfg',\n                and 'transformation_3d_flow'. For details on the values of\n                the keys see `mmdet/datasets/pipelines/formatting.py:Collect`.\n            gt_bboxes (list[torch.Tensor]): Ground truth bboxes for each image\n                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.\n            gt_labels (list[torch.Tensor]): class indices for each\n                2d bounding box.\n            gt_bboxes_ignore (list[torch.Tensor]): specify which\n                2d bounding boxes can be ignored when computing the loss.\n            gt_masks (torch.Tensor): true segmentation masks for each\n                2d bbox, used if the architecture supports a segmentation task.\n            proposals: override rpn proposals (2d) with custom proposals.\n                Use when `with_rpn` is False.\n            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,\n                not supported yet.\n            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): 3d gt bboxes.\n            gt_labels_3d (list[torch.Tensor]): gt class labels for 3d bboxes.\n            pts_semantic_mask (list[torch.Tensor]): point-wise semantic\n                label of each batch.\n            pts_instance_mask (list[torch.Tensor]): point-wise instance\n                label of each batch.\n\n        Returns:\n            dict[str, torch.Tensor]: a dictionary of loss components.\n        "
        if points is None:
            x = self.extract_img_feat(img)
            losses = dict()
            if self.with_img_rpn:
                proposal_cfg = self.train_cfg.get('img_rpn_proposal', self.test_cfg.img_rpn)
                (rpn_losses, proposal_list) = self.img_rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals
            roi_losses = self.img_roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            bboxes_2d = self.extract_bboxes_2d(img, img_metas, bboxes_2d=bboxes_2d, **kwargs)
            points = torch.stack(points)
            (seeds_3d, seed_3d_features, seed_indices) = self.extract_pts_feat(points)
            (img_features, masks) = self.fusion_layer(img, bboxes_2d, seeds_3d, img_metas)
            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            (batch_size, img_feat_size) = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1, -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1, -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)
            img_features = self.img_mlp(img_features)
            fused_features = torch.cat([seed_3d_features, img_features], dim=1)
            feat_dict_joint = dict(seed_points=seeds_3d, seed_features=fused_features, seed_indices=seed_indices)
            feat_dict_pts = dict(seed_points=seeds_3d, seed_features=seed_3d_features, seed_indices=seed_indices)
            feat_dict_img = dict(seed_points=seeds_3d, seed_features=img_features, seed_indices=seed_indices)
            loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask, img_metas)
            bbox_preds_joints = self.pts_bbox_head_joint(feat_dict_joint, self.train_cfg.pts.sample_mod)
            bbox_preds_pts = self.pts_bbox_head_pts(feat_dict_pts, self.train_cfg.pts.sample_mod)
            bbox_preds_img = self.pts_bbox_head_img(feat_dict_img, self.train_cfg.pts.sample_mod)
            losses_towers = []
            losses_joint = self.pts_bbox_head_joint.loss(bbox_preds_joints, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_pts = self.pts_bbox_head_pts.loss(bbox_preds_pts, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_img = self.pts_bbox_head_img.loss(bbox_preds_img, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_towers.append(losses_joint)
            losses_towers.append(losses_pts)
            losses_towers.append(losses_img)
            combined_losses = dict()
            for loss_term in losses_joint:
                if 'loss' in loss_term:
                    combined_losses[loss_term] = 0
                    for i in range(len(losses_towers)):
                        combined_losses[loss_term] += losses_towers[i][loss_term] * self.loss_weights[i]
                else:
                    combined_losses[loss_term] = losses_towers[0][loss_term]
            return combined_losses

    def forward_test(self, points=None, img_metas=None, img=None, bboxes_2d=None, **kwargs):
        if False:
            while True:
                i = 10
        'Forwarding of test for image branch pretrain or stage 2 train.\n\n        Args:\n            points (list[list[torch.Tensor]], optional): the outer\n                list indicates test-time augmentations and the inner\n                list contains all points in the batch, where each Tensor\n                should have a shape NxC. Defaults to None.\n            img_metas (list[list[dict]], optional): the outer list\n                indicates test-time augs (multiscale, flip, etc.)\n                and the inner list indicates images in a batch.\n                Defaults to None.\n            img (list[list[torch.Tensor]], optional): the outer\n                list indicates test-time augmentations and inner Tensor\n                should have a shape NxCxHxW, which contains all images\n                in the batch. Defaults to None. Defaults to None.\n            bboxes_2d (list[list[torch.Tensor]], optional):\n                Provided 2d bboxes, not supported yet. Defaults to None.\n\n        Returns:\n            list[list[torch.Tensor]]|list[dict]: Predicted 2d or 3d boxes.\n        '
        if points is None:
            for (var, name) in [(img, 'img'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(f'{name} must be a list, but got {type(var)}')
            num_augs = len(img)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(img)}) != num of image meta ({len(img_metas)})')
            if num_augs == 1:
                if 'proposals' in kwargs:
                    kwargs['proposals'] = kwargs['proposals'][0]
                return self.simple_test_img_only(img=img[0], img_metas=img_metas[0], **kwargs)
            else:
                assert img[0].size(0) == 1, f'aug test does not support inference with batch size {img[0].size(0)}'
                assert 'proposals' not in kwargs
                return self.aug_test_img_only(img=img, img_metas=img_metas, **kwargs)
        else:
            for (var, name) in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(points), len(img_metas)))
            if num_augs == 1:
                return self.simple_test(points[0], img_metas[0], img[0], bboxes_2d=bboxes_2d[0] if bboxes_2d is not None else None, **kwargs)
            else:
                return self.aug_test(points, img_metas, img, bboxes_2d, **kwargs)

    def simple_test_img_only(self, img, img_metas, proposals=None, rescale=False):
        if False:
            i = 10
            return i + 15
        'Test without augmentation, image network pretrain. May refer to\n        `<https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py>`_.\n\n        Args:\n            img (torch.Tensor): Should have a shape NxCxHxW, which contains\n                all images in the batch.\n            img_metas (list[dict]):\n            proposals (list[Tensor], optional): override rpn proposals\n                with custom proposals. Defaults to None.\n            rescale (bool, optional): Whether or not rescale bboxes to the\n                original shape of input image. Defaults to False.\n\n        Returns:\n            list[list[torch.Tensor]]: Predicted 2d boxes.\n        '
        assert self.with_img_bbox, 'Img bbox head must be implemented.'
        assert self.with_img_backbone, 'Img backbone must be implemented.'
        assert self.with_img_rpn, 'Img rpn must be implemented.'
        assert self.with_img_roi_head, 'Img roi head must be implemented.'
        x = self.extract_img_feat(img)
        if proposals is None:
            proposal_list = self.img_rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        ret = self.img_roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        return ret

    def simple_test(self, points=None, img_metas=None, img=None, bboxes_2d=None, rescale=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Test without augmentation, stage 2.\n\n        Args:\n            points (list[torch.Tensor], optional): Elements in the list\n                should have a shape NxC, the list indicates all point-clouds\n                in the batch. Defaults to None.\n            img_metas (list[dict], optional): List indicates\n                images in a batch. Defaults to None.\n            img (torch.Tensor, optional): Should have a shape NxCxHxW,\n                which contains all images in the batch. Defaults to None.\n            bboxes_2d (list[torch.Tensor], optional):\n                Provided 2d bboxes, not supported yet. Defaults to None.\n            rescale (bool, optional): Whether or not rescale bboxes.\n                Defaults to False.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        bboxes_2d = self.extract_bboxes_2d(img, img_metas, train=False, bboxes_2d=bboxes_2d, **kwargs)
        points = torch.stack(points)
        (seeds_3d, seed_3d_features, seed_indices) = self.extract_pts_feat(points)
        (img_features, masks) = self.fusion_layer(img, bboxes_2d, seeds_3d, img_metas)
        inds = sample_valid_seeds(masks, self.num_sampled_seed)
        (batch_size, img_feat_size) = img_features.shape[:2]
        pts_feat_size = seed_3d_features.shape[1]
        inds_img = inds.view(batch_size, 1, -1).expand(-1, img_feat_size, -1)
        img_features = img_features.gather(-1, inds_img)
        inds = inds % inds.shape[1]
        inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
        seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
        inds_seed_feats = inds.view(batch_size, 1, -1).expand(-1, pts_feat_size, -1)
        seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
        seed_indices = seed_indices.gather(1, inds)
        img_features = self.img_mlp(img_features)
        fused_features = torch.cat([seed_3d_features, img_features], dim=1)
        feat_dict = dict(seed_points=seeds_3d, seed_features=fused_features, seed_indices=seed_indices)
        bbox_preds = self.pts_bbox_head_joint(feat_dict, self.test_cfg.pts.sample_mod)
        bbox_list = self.pts_bbox_head_joint.get_bboxes(points, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for (bboxes, scores, labels) in bbox_list]
        return bbox_results

    def aug_test_img_only(self, img, img_metas, rescale=False):
        if False:
            i = 10
            return i + 15
        'Test function with augmentation, image network pretrain. May refer\n        to `<https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py>`_.\n\n        Args:\n            img (list[list[torch.Tensor]], optional): the outer\n                list indicates test-time augmentations and inner Tensor\n                should have a shape NxCxHxW, which contains all images\n                in the batch. Defaults to None. Defaults to None.\n            img_metas (list[list[dict]], optional): the outer list\n                indicates test-time augs (multiscale, flip, etc.)\n                and the inner list indicates images in a batch.\n                Defaults to None.\n            rescale (bool, optional): Whether or not rescale bboxes to the\n                original shape of input image. If rescale is False, then\n                returned bboxes and masks will fit the scale of imgs[0].\n                Defaults to None.\n\n        Returns:\n            list[list[torch.Tensor]]: Predicted 2d boxes.\n        '
        assert self.with_img_bbox, 'Img bbox head must be implemented.'
        assert self.with_img_backbone, 'Img backbone must be implemented.'
        assert self.with_img_rpn, 'Img rpn must be implemented.'
        assert self.with_img_roi_head, 'Img roi head must be implemented.'
        x = self.extract_img_feats(img)
        proposal_list = self.img_rpn_head.aug_test_rpn(x, img_metas)
        return self.img_roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, points=None, img_metas=None, imgs=None, bboxes_2d=None, rescale=False, **kwargs):
        if False:
            return 10
        'Test function with augmentation, stage 2.\n\n        Args:\n            points (list[list[torch.Tensor]], optional): the outer\n                list indicates test-time augmentations and the inner\n                list contains all points in the batch, where each Tensor\n                should have a shape NxC. Defaults to None.\n            img_metas (list[list[dict]], optional): the outer list\n                indicates test-time augs (multiscale, flip, etc.)\n                and the inner list indicates images in a batch.\n                Defaults to None.\n            imgs (list[list[torch.Tensor]], optional): the outer\n                list indicates test-time augmentations and inner Tensor\n                should have a shape NxCxHxW, which contains all images\n                in the batch. Defaults to None. Defaults to None.\n            bboxes_2d (list[list[torch.Tensor]], optional):\n                Provided 2d bboxes, not supported yet. Defaults to None.\n            rescale (bool, optional): Whether or not rescale bboxes.\n                Defaults to False.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_pts_feats(points_cat, img_metas)
        aug_bboxes = []
        for (x, pts_cat, img_meta, bbox_2d, img) in zip(feats, points_cat, img_metas, bboxes_2d, imgs):
            bbox_2d = self.extract_bboxes_2d(img, img_metas, train=False, bboxes_2d=bbox_2d, **kwargs)
            (seeds_3d, seed_3d_features, seed_indices) = x
            (img_features, masks) = self.fusion_layer(img, bbox_2d, seeds_3d, img_metas)
            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            (batch_size, img_feat_size) = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1, -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1, -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)
            img_features = self.img_mlp(img_features)
            fused_features = torch.cat([seed_3d_features, img_features], dim=1)
            feat_dict = dict(seed_points=seeds_3d, seed_features=fused_features, seed_indices=seed_indices)
            bbox_preds = self.pts_bbox_head_joint(feat_dict, self.test_cfg.pts.sample_mod)
            bbox_list = self.pts_bbox_head_joint.get_bboxes(pts_cat, bbox_preds, img_metas, rescale=rescale)
            bbox_list = [dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for (bboxes, scores, labels) in bbox_list]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.bbox_head.test_cfg)
        return [merged_bboxes]
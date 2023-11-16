import torch
from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.core.bbox.structures.utils import get_proj_mat_by_coord_type
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class ImVoxelNet(BaseDetector):
    """`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_.

    Args:
        backbone (dict): Config of the backbone.
        neck (dict): Config of the 2d neck.
        neck_3d (dict): Config of the 3d neck.
        bbox_head (dict): Config of the head.
        prior_generator (dict): Config of the prior generator.
        n_voxels (tuple[int]): Number of voxels for x, y, and z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self, backbone, neck, neck_3d, bbox_head, prior_generator, n_voxels, coord_type, train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.coord_type = coord_type
        self.prior_generator = build_prior_generator(prior_generator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img, img_metas):
        if False:
            i = 10
            return i + 15
        'Extract 3d features from the backbone -> fpn -> 3d projection.\n\n        -> 3d neck -> bbox_head.\n\n        Args:\n            img (torch.Tensor): Input images of shape (N, C_in, H, W).\n            img_metas (list): Image metas.\n\n        Returns:\n            Tuple:\n             - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).\n             - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).\n        '
        x = self.backbone(img)
        x = self.neck(x)[0]
        points = self.prior_generator.grid_anchors([self.n_voxels[::-1]], device=img.device)[0][:, :3]
        (volumes, valid_preds) = ([], [])
        for (feature, img_meta) in zip(x, img_metas):
            img_scale_factor = points.new_tensor(img_meta['scale_factor'][:2]) if 'scale_factor' in img_meta.keys() else 1
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = points.new_tensor(img_meta['img_crop_offset']) if 'img_crop_offset' in img_meta.keys() else 0
            proj_mat = points.new_tensor(get_proj_mat_by_coord_type(img_meta, self.coord_type))
            volume = point_sample(img_meta, img_features=feature[None, ...], points=points, proj_mat=points.new_tensor(proj_mat), coord_type=self.coord_type, img_scale_factor=img_scale_factor, img_crop_offset=img_crop_offset, img_flip=img_flip, img_pad_shape=img.shape[-2:], img_shape=img_meta['img_shape'][:2], aligned=False)
            volumes.append(volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            valid_preds.append(~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        x = torch.stack(volumes)
        x = self.neck_3d(x)
        x = self.bbox_head(x)
        return (x, torch.stack(valid_preds).float())

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        if False:
            return 10
        'Forward of training.\n\n        Args:\n            img (torch.Tensor): Input images of shape (N, C_in, H, W).\n            img_metas (list): Image metas.\n            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.\n            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.\n\n        Returns:\n            dict[str, torch.Tensor]: A dictionary of loss components.\n        '
        (x, valid_preds) = self.extract_feat(img, img_metas)
        if self.coord_type == 'DEPTH':
            x += (valid_preds,)
        losses = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Forward of testing.\n\n        Args:\n            img (torch.Tensor): Input images of shape (N, C_in, H, W).\n            img_metas (list): Image metas.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        if False:
            while True:
                i = 10
        'Test without augmentations.\n\n        Args:\n            img (torch.Tensor): Input images of shape (N, C_in, H, W).\n            img_metas (list): Image metas.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        (x, valid_preds) = self.extract_feat(img, img_metas)
        if self.coord_type == 'DEPTH':
            x += (valid_preds,)
        bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        bbox_results = [bbox3d2result(det_bboxes, det_scores, det_labels) for (det_bboxes, det_scores, det_labels) in bbox_list]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        if False:
            return 10
        'Test with augmentations.\n\n        Args:\n            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).\n            img_metas (list): Image metas.\n\n        Returns:\n            list[dict]: Predicted 3d boxes.\n        '
        raise NotImplementedError
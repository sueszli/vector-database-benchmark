import warnings
from os import path as osp
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.core import CameraInstance3DBoxes, bbox3d2result, show_multi_modality_result
from mmdet.models.detectors import SingleStageDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class SingleStageMono3DDetector(SingleStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if False:
            for i in range(10):
                print('nop')
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feats(self, imgs):
        if False:
            i = 10
            return i + 15
        'Directly extract features from the backbone+neck.'
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, attr_labels=None, gt_bboxes_ignore=None):
        if False:
            while True:
                i = 10
        "\n        Args:\n            img (Tensor): Input images of shape (N, C, H, W).\n                Typically these should be mean centered and std scaled.\n            img_metas (list[dict]): A List of image info dict where each dict\n                has: 'img_shape', 'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                :class:`mmdet.datasets.pipelines.Collect`.\n            gt_bboxes (list[Tensor]): Each item are the truth boxes for each\n                image in [tl_x, tl_y, br_x, br_y] format.\n            gt_labels (list[Tensor]): Class indices corresponding to each box\n            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for\n                each image in [x, y, z, x_size, y_size, z_size, yaw, vx, vy]\n                format.\n            gt_labels_3d (list[Tensor]): 3D class indices corresponding to\n                each box.\n            centers2d (list[Tensor]): Projected 3D centers onto 2D images.\n            depths (list[Tensor]): Depth of projected centers on 2D images.\n            attr_labels (list[Tensor], optional): Attribute indices\n                corresponding to each box\n            gt_bboxes_ignore (list[Tensor]): Specify which bounding\n                boxes can be ignored when computing the loss.\n\n        Returns:\n            dict[str, Tensor]: A dictionary of loss components.\n        "
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, depths, attr_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        if False:
            print('Hello World!')
        'Test function without test time augmentation.\n\n        Args:\n            imgs (list[torch.Tensor]): List of multiple images\n            img_metas (list[dict]): List of image information.\n            rescale (bool, optional): Whether to rescale the results.\n                Defaults to False.\n\n        Returns:\n            list[list[np.ndarray]]: BBox results of each image and classes.\n                The outer list corresponds to each image. The inner list\n                corresponds to each class.\n        '
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [bbox2result(bboxes2d, labels, self.bbox_head.num_classes) for (bboxes, scores, labels, attrs, bboxes2d) in bbox_outputs]
            bbox_outputs = [bbox_outputs[0][:-1]]
        bbox_img = [bbox3d2result(bboxes, scores, labels, attrs) for (bboxes, scores, labels, attrs) in bbox_outputs]
        bbox_list = [dict() for i in range(len(img_metas))]
        for (result_dict, img_bbox) in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for (result_dict, img_bbox2d) in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        if False:
            for i in range(10):
                print('nop')
        'Test function with test time augmentation.'
        feats = self.extract_feats(imgs)
        outs_list = [self.bbox_head(x) for x in feats]
        for (i, img_meta) in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        outs_list[i][j][k] = torch.flip(outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi
        merged_outs = []
        for i in range(len(outs_list[0])):
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                avg_feats = torch.mean(torch.cat([x[i][j] for x in outs_list]), dim=0, keepdim=True)
                if i == 1:
                    avg_feats[:, 6:, :, :] = outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)
        bbox_outputs = self.bbox_head.get_bboxes(*merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [bbox2result(bboxes2d, labels, self.bbox_head.num_classes) for (bboxes, scores, labels, attrs, bboxes2d) in bbox_outputs]
            bbox_outputs = [bbox_outputs[0][:-1]]
        bbox_img = [bbox3d2result(bboxes, scores, labels, attrs) for (bboxes, scores, labels, attrs) in bbox_outputs]
        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])
        return [bbox_list]

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
        if False:
            i = 10
            return i + 15
        'Results visualization.\n\n        Args:\n            data (list[dict]): Input images and the information of the sample.\n            result (list[dict]): Prediction results.\n            out_dir (str): Output directory of visualization result.\n            show (bool, optional): Determines whether you are\n                going to show result by open3d.\n                Defaults to False.\n            TODO: implement score_thr of single_stage_mono3d.\n            score_thr (float, optional): Score threshold of bounding boxes.\n                Default to None.\n                Not implemented yet, but it is here for unification.\n        '
        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id]['filename']
                cam2img = data['img_metas'][0]._data[0][batch_id]['cam2img']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['cam2img']
            else:
                ValueError(f"Unsupported data type {type(data['img_metas'][0])} for visualization!")
            img = mmcv.imread(img_filename)
            file_name = osp.split(img_filename)[-1].split('.')[0]
            assert out_dir is not None, 'Expect out_dir, got none.'
            pred_bboxes = result[batch_id]['img_bbox']['boxes_3d']
            assert isinstance(pred_bboxes, CameraInstance3DBoxes), f'unsupported predicted bbox type {type(pred_bboxes)}'
            show_multi_modality_result(img, None, pred_bboxes, cam2img, out_dir, file_name, 'camera', show=show)
import numpy as np
import torch
from torch.nn import functional as F
from mmdet.core.bbox.builder import BBOX_CODERS
from .fcos3d_bbox_coder import FCOS3DBBoxCoder

@BBOX_CODERS.register_module()
class PGDBBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for PGD."""

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        if False:
            print('Hello World!')
        pass

    def decode_2d(self, bbox, scale, stride, max_regress_range, training, pred_keypoints=False, pred_bbox2d=True):
        if False:
            print('Hello World!')
        'Decode regressed 2D attributes.\n\n        Args:\n            bbox (torch.Tensor): Raw bounding box predictions in shape\n                [N, C, H, W].\n            scale (tuple[`Scale`]): Learnable scale parameters.\n            stride (int): Stride for a specific feature level.\n            max_regress_range (int): Maximum regression range for a specific\n                feature level.\n            training (bool): Whether the decoding is in the training\n                procedure.\n            pred_keypoints (bool, optional): Whether to predict keypoints.\n                Defaults to False.\n            pred_bbox2d (bool, optional): Whether to predict 2D bounding\n                boxes. Defaults to False.\n\n        Returns:\n            torch.Tensor: Decoded boxes.\n        '
        clone_bbox = bbox.clone()
        if pred_keypoints:
            scale_kpts = scale[3]
            bbox[:, self.bbox_code_size:self.bbox_code_size + 16] = torch.tanh(scale_kpts(clone_bbox[:, self.bbox_code_size:self.bbox_code_size + 16]).float())
        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()
        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    bbox[:, self.bbox_code_size:self.bbox_code_size + 16] *= max_regress_range
                if pred_bbox2d:
                    bbox[:, -4:] *= stride
        elif pred_bbox2d:
            bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox

    def decode_prob_depth(self, depth_cls_preds, depth_range, depth_unit, division, num_depth_cls):
        if False:
            for i in range(10):
                print('nop')
        "Decode probabilistic depth map.\n\n        Args:\n            depth_cls_preds (torch.Tensor): Depth probabilistic map in shape\n                [..., self.num_depth_cls] (raw output before softmax).\n            depth_range (tuple[float]): Range of depth estimation.\n            depth_unit (int): Unit of depth range division.\n            division (str): Depth division method. Options include 'uniform',\n                'linear', 'log', 'loguniform'.\n            num_depth_cls (int): Number of depth classes.\n\n        Returns:\n            torch.Tensor: Decoded probabilistic depth estimation.\n        "
        if division == 'uniform':
            depth_multiplier = depth_unit * depth_cls_preds.new_tensor(list(range(num_depth_cls))).reshape([1, -1])
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) * depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'linear':
            split_pts = depth_cls_preds.new_tensor(list(range(num_depth_cls))).reshape([1, -1])
            depth_multiplier = depth_range[0] + (depth_range[1] - depth_range[0]) / (num_depth_cls * (num_depth_cls - 1)) * (split_pts * (split_pts + 1))
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) * depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'log':
            split_pts = depth_cls_preds.new_tensor(list(range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            depth_multiplier = (np.log(start) + split_pts * np.log(end / start) / (num_depth_cls - 1)).exp()
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) * depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'loguniform':
            split_pts = depth_cls_preds.new_tensor(list(range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            log_multiplier = np.log(start) + split_pts * np.log(end / start) / (num_depth_cls - 1)
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) * log_multiplier).sum(dim=-1).exp()
            return prob_depth_preds
        else:
            raise NotImplementedError
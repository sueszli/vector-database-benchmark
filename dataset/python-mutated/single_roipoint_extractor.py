import torch
from mmcv import ops
from torch import nn as nn
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import ROI_EXTRACTORS

@ROI_EXTRACTORS.register_module()
class Single3DRoIPointExtractor(nn.Module):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict): The config of roi layer.
    """

    def __init__(self, roi_layer=None):
        if False:
            while True:
                i = 10
        super(Single3DRoIPointExtractor, self).__init__()
        self.roi_layer = self.build_roi_layers(roi_layer)

    def build_roi_layers(self, layer_cfg):
        if False:
            while True:
                i = 10
        'Build roi layers using `layer_cfg`'
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = layer_cls(**cfg)
        return roi_layers

    def forward(self, feats, coordinate, batch_inds, rois):
        if False:
            print('Hello World!')
        'Extract point-wise roi features.\n\n        Args:\n            feats (torch.FloatTensor): Point-wise features with\n                shape (batch, npoints, channels) for pooling.\n            coordinate (torch.FloatTensor): Coordinate of each point.\n            batch_inds (torch.LongTensor): Indicate the batch of each point.\n            rois (torch.FloatTensor): Roi boxes with batch indices.\n\n        Returns:\n            torch.FloatTensor: Pooled features\n        '
        rois = rois[..., 1:]
        rois = rois.view(batch_inds, -1, rois.shape[-1])
        with torch.no_grad():
            (pooled_roi_feat, pooled_empty_flag) = self.roi_layer(coordinate, feats, rois)
            roi_center = rois[:, :, 0:3]
            pooled_roi_feat[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            pooled_roi_feat = pooled_roi_feat.view(-1, pooled_roi_feat.shape[-2], pooled_roi_feat.shape[-1])
            pooled_roi_feat[:, :, 0:3] = rotation_3d_in_axis(pooled_roi_feat[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6], axis=2)
            pooled_roi_feat[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_roi_feat
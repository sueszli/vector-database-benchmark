import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from mmcv.runner import force_fp32
from torch import nn
from .. import builder
from ..builder import VOXEL_ENCODERS
from .utils import VFELayer, get_paddings_indicator

@VOXEL_ENCODERS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int, optional): Number of features to use. Default: 4.
    """

    def __init__(self, num_features=4):
        if False:
            return 10
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        if False:
            while True:
                i = 10
        'Forward function.\n\n        Args:\n            features (torch.Tensor): Point features in shape\n                (N, M, 3(4)). N is the number of voxels and M is the maximum\n                number of points inside a single voxel.\n            num_points (torch.Tensor): Number of points in each voxel,\n                 shape (N, ).\n            coors (torch.Tensor): Coordinates of voxels.\n\n        Returns:\n            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))\n        '
        points_mean = features[:, :, :self.num_features].sum(dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()

@VOXEL_ENCODERS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self, voxel_size=(0.2, 0.2, 4), point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        if False:
            return 10
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        if False:
            print('Hello World!')
        'Forward function.\n\n        Args:\n            features (torch.Tensor): Point features in shape\n                (N, 3(4)). N is the number of points.\n            coors (torch.Tensor): Coordinates of voxels.\n\n        Returns:\n            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).\n                M is the number of voxels.\n        '
        (features, features_coors) = self.scatter(features, coors)
        return (features, features_coors)

@VOXEL_ENCODERS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance of
            points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion
            layer used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self, in_channels=4, feat_channels=[], with_distance=False, with_cluster_center=False, with_voxel_center=False, voxel_size=(0.2, 0.2, 4), point_cloud_range=(0, -40, -3, 70.4, 40, 1), norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01), mode='max', fusion_layer=None, return_point_feats=False):
        if False:
            for i in range(10):
                print('nop')
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            (norm_name, norm_layer) = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(nn.Sequential(nn.Linear(in_filters, out_filters, bias=False), norm_layer, nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range, mode != 'max')
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        if False:
            i = 10
            return i + 15
        'Map voxel features to its corresponding points.\n\n        Args:\n            pts_coors (torch.Tensor): Voxel coordinate of each point.\n            voxel_mean (torch.Tensor): Voxel features to be mapped.\n            voxel_coors (torch.Tensor): Coordinates of valid voxels\n\n        Returns:\n            torch.Tensor: Features or centers of each point.\n        '
        canvas_z = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        indices = voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x + voxel_coors[:, 1] * canvas_y * canvas_x + voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3]
        canvas[indices.long()] = torch.arange(start=0, end=voxel_mean.size(0), device=voxel_mean.device)
        voxel_index = pts_coors[:, 0] * canvas_z * canvas_y * canvas_x + pts_coors[:, 1] * canvas_y * canvas_x + pts_coors[:, 2] * canvas_x + pts_coors[:, 3]
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    @force_fp32(out_fp16=True)
    def forward(self, features, coors, points=None, img_feats=None, img_metas=None):
        if False:
            i = 10
            return i + 15
        'Forward functions.\n\n        Args:\n            features (torch.Tensor): Features of voxels, shape is NxC.\n            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).\n            points (list[torch.Tensor], optional): Raw points used to guide the\n                multi-modality fusion. Defaults to None.\n            img_feats (list[torch.Tensor], optional): Image features used for\n                multi-modality fusion. Defaults to None.\n            img_metas (dict, optional): [description]. Defaults to None.\n\n        Returns:\n            tuple: If `return_point_feats` is False, returns voxel features and\n                its coordinates. If `return_point_feats` is True, returns\n                feature of each points inside voxels.\n        '
        features_ls = [features]
        if self._with_cluster_center:
            (voxel_mean, mean_coors) = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(coors, voxel_mean, mean_coors)
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)
        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
        for (i, vfe) in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if i == len(self.vfe_layers) - 1 and self.fusion_layer is not None and (img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)
            (voxel_feats, voxel_coors) = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                feat_per_point = self.map_voxel_center_to_point(coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        return (voxel_feats, voxel_coors)

@VOXEL_ENCODERS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance
            of points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance to
            center of voxel for each points inside a voxel. Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points inside a
            voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion layer
            used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the
            features of each points. Defaults to False.
    """

    def __init__(self, in_channels=4, feat_channels=[], with_distance=False, with_cluster_center=False, with_voxel_center=False, voxel_size=(0.2, 0.2, 4), point_cloud_range=(0, -40, -3, 70.4, 40, 1), norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01), mode='max', fusion_layer=None, return_point_feats=False):
        if False:
            i = 10
            return i + 15
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            if i == len(feat_channels) - 2:
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(VFELayer(in_filters, out_filters, norm_cfg=norm_cfg, max_out=max_out, cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors, img_feats=None, img_metas=None):
        if False:
            print('Hello World!')
        'Forward functions.\n\n        Args:\n            features (torch.Tensor): Features of voxels, shape is MxNxC.\n            num_points (torch.Tensor): Number of points in each voxel.\n            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).\n            img_feats (list[torch.Tensor], optional): Image features used for\n                multi-modality fusion. Defaults to None.\n            img_metas (dict, optional): [description]. Defaults to None.\n\n        Returns:\n            tuple: If `return_point_feats` is False, returns voxel features and\n                its coordinates. If `return_point_feats` is True, returns\n                feature of each points inside voxels.\n        '
        features_ls = [features]
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (coors[:, 1].type_as(features).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        voxel_feats = torch.cat(features_ls, dim=-1)
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)
        for (i, vfe) in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
        if self.fusion_layer is not None and img_feats is not None:
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats, coors, img_feats, img_metas)
        return voxel_feats

    def fusion_with_mask(self, features, mask, voxel_feats, coors, img_feats, img_metas):
        if False:
            print('Hello World!')
        'Fuse image and point features with mask.\n\n        Args:\n            features (torch.Tensor): Features of voxel, usually it is the\n                values of points in voxels.\n            mask (torch.Tensor): Mask indicates valid features in each voxel.\n            voxel_feats (torch.Tensor): Features of voxels.\n            coors (torch.Tensor): Coordinates of each single voxel.\n            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.\n            img_metas (list(dict)): Meta information of image and points.\n\n        Returns:\n            torch.Tensor: Fused features of each voxel.\n        '
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = coors[:, 0] == i
            points.append(features[single_mask][mask[single_mask]])
        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)
        voxel_canvas = voxel_feats.new_zeros(size=(voxel_feats.size(0), voxel_feats.size(1), point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]
        return out
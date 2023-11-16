import numpy as np
import torch
from torch.nn import functional as F
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

@BBOX_CODERS.register_module()
class MonoFlexCoder(BaseBBoxCoder):
    """Bbox Coder for MonoFlex.

    Args:
        depth_mode (str): The mode for depth calculation.
            Available options are "linear", "inv_sigmoid", and "exp".
        base_depth (tuple[float]): References for decoding box depth.
        depth_range (list): Depth range of predicted depth.
        combine_depth (bool): Whether to use combined depth (direct depth
            and depth from keypoints) or use direct depth only.
        uncertainty_range (list): Uncertainty range of predicted depth.
        base_dims (tuple[tuple[float]]): Dimensions mean and std of decode bbox
            dimensions [l, h, w] for each category.
        dims_mode (str): The mode for dimension calculation.
            Available options are "linear" and "exp".
        multibin (bool): Whether to use multibin representation.
        num_dir_bins (int): Number of Number of bins to encode
            direction angle.
        bin_centers (list[float]): Local yaw centers while using multibin
            representations.
        bin_margin (float): Margin of multibin representations.
        code_size (int): The dimension of boxes to be encoded.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-3.
    """

    def __init__(self, depth_mode, base_depth, depth_range, combine_depth, uncertainty_range, base_dims, dims_mode, multibin, num_dir_bins, bin_centers, bin_margin, code_size, eps=0.001):
        if False:
            i = 10
            return i + 15
        super(MonoFlexCoder, self).__init__()
        self.depth_mode = depth_mode
        self.base_depth = base_depth
        self.depth_range = depth_range
        self.combine_depth = combine_depth
        self.uncertainty_range = uncertainty_range
        self.base_dims = base_dims
        self.dims_mode = dims_mode
        self.multibin = multibin
        self.num_dir_bins = num_dir_bins
        self.bin_centers = bin_centers
        self.bin_margin = bin_margin
        self.bbox_code_size = code_size
        self.eps = eps

    def encode(self, gt_bboxes_3d):
        if False:
            print('Hello World!')
        'Encode ground truth to prediction targets.\n\n        Args:\n            gt_bboxes_3d (`BaseInstance3DBoxes`): Ground truth 3D bboxes.\n                shape: (N, 7).\n\n        Returns:\n            torch.Tensor: Targets of orientations.\n        '
        local_yaw = gt_bboxes_3d.local_yaw
        encode_local_yaw = local_yaw.new_zeros([local_yaw.shape[0], self.num_dir_bins * 2])
        bin_size = 2 * np.pi / self.num_dir_bins
        margin_size = bin_size * self.bin_margin
        bin_centers = local_yaw.new_tensor(self.bin_centers)
        range_size = bin_size / 2 + margin_size
        offsets = local_yaw.unsqueeze(1) - bin_centers.unsqueeze(0)
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi
        for i in range(self.num_dir_bins):
            offset = offsets[:, i]
            inds = abs(offset) < range_size
            encode_local_yaw[inds, i] = 1
            encode_local_yaw[inds, i + self.num_dir_bins] = offset[inds]
        orientation_target = encode_local_yaw
        return orientation_target

    def decode(self, bbox, base_centers2d, labels, downsample_ratio, cam2imgs):
        if False:
            while True:
                i = 10
        "Decode bounding box regression into 3D predictions.\n\n        Args:\n            bbox (Tensor): Raw bounding box predictions for each\n                predict center2d point.\n                shape: (N, C)\n            base_centers2d (torch.Tensor): Base centers2d for 3D bboxes.\n                shape: (N, 2).\n            labels (Tensor): Batch predict class label for each predict\n                center2d point.\n                shape: (N, )\n            downsample_ratio (int): The stride of feature map.\n            cam2imgs (Tensor): Batch images' camera intrinsic matrix.\n                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)\n\n        Return:\n            dict: The 3D prediction dict decoded from regression map.\n            the dict has components below:\n                - bboxes2d (torch.Tensor): Decoded [x1, y1, x2, y2] format\n                    2D bboxes.\n                - dimensions (torch.Tensor): Decoded dimensions for each\n                    object.\n                - offsets2d (torch.Tenosr): Offsets between base centers2d\n                    and real centers2d.\n                - direct_depth (torch.Tensor): Decoded directly regressed\n                    depth.\n                - keypoints2d (torch.Tensor): Keypoints of each projected\n                    3D box on image.\n                - keypoints_depth (torch.Tensor): Decoded depth from keypoints.\n                - combined_depth (torch.Tensor): Combined depth using direct\n                    depth and keypoints depth with depth uncertainty.\n                - orientations (torch.Tensor): Multibin format orientations\n                    (local yaw) for each objects.\n        "
        pred_bboxes2d = bbox[:, 0:4]
        pred_bboxes2d = self.decode_bboxes2d(pred_bboxes2d, base_centers2d)
        pred_offsets2d = bbox[:, 4:6]
        pred_dimensions_offsets3d = bbox[:, 29:32]
        pred_orientations = torch.cat((bbox[:, 32:40], bbox[:, 40:48]), dim=1)
        pred_keypoints_depth_uncertainty = bbox[:, 26:29]
        pred_direct_depth_uncertainty = bbox[:, 49:50].squeeze(-1)
        pred_keypoints2d = bbox[:, 6:26].reshape(-1, 10, 2)
        pred_direct_depth_offsets = bbox[:, 48:49].squeeze(-1)
        pred_dimensions = self.decode_dims(labels, pred_dimensions_offsets3d)
        pred_direct_depth = self.decode_direct_depth(pred_direct_depth_offsets)
        pred_keypoints_depth = self.keypoints2depth(pred_keypoints2d, pred_dimensions, cam2imgs, downsample_ratio)
        pred_direct_depth_uncertainty = torch.clamp(pred_direct_depth_uncertainty, self.uncertainty_range[0], self.uncertainty_range[1])
        pred_keypoints_depth_uncertainty = torch.clamp(pred_keypoints_depth_uncertainty, self.uncertainty_range[0], self.uncertainty_range[1])
        if self.combine_depth:
            pred_depth_uncertainty = torch.cat((pred_direct_depth_uncertainty.unsqueeze(-1), pred_keypoints_depth_uncertainty), dim=1).exp()
            pred_depth = torch.cat((pred_direct_depth.unsqueeze(-1), pred_keypoints_depth), dim=1)
            pred_combined_depth = self.combine_depths(pred_depth, pred_depth_uncertainty)
        else:
            pred_combined_depth = None
        preds = dict(bboxes2d=pred_bboxes2d, dimensions=pred_dimensions, offsets2d=pred_offsets2d, keypoints2d=pred_keypoints2d, orientations=pred_orientations, direct_depth=pred_direct_depth, keypoints_depth=pred_keypoints_depth, combined_depth=pred_combined_depth, direct_depth_uncertainty=pred_direct_depth_uncertainty, keypoints_depth_uncertainty=pred_keypoints_depth_uncertainty)
        return preds

    def decode_direct_depth(self, depth_offsets):
        if False:
            return 10
        'Transform depth offset to directly regressed depth.\n\n        Args:\n            depth_offsets (torch.Tensor): Predicted depth offsets.\n                shape: (N, )\n\n        Return:\n            torch.Tensor: Directly regressed depth.\n                shape: (N, )\n        '
        if self.depth_mode == 'exp':
            direct_depth = depth_offsets.exp()
        elif self.depth_mode == 'linear':
            base_depth = depth_offsets.new_tensor(self.base_depth)
            direct_depth = depth_offsets * base_depth[1] + base_depth[0]
        elif self.depth_mode == 'inv_sigmoid':
            direct_depth = 1 / torch.sigmoid(depth_offsets) - 1
        else:
            raise ValueError
        if self.depth_range is not None:
            direct_depth = torch.clamp(direct_depth, min=self.depth_range[0], max=self.depth_range[1])
        return direct_depth

    def decode_location(self, base_centers2d, offsets2d, depths, cam2imgs, downsample_ratio, pad_mode='default'):
        if False:
            while True:
                i = 10
        "Retrieve object location.\n\n        Args:\n            base_centers2d (torch.Tensor): predicted base centers2d.\n                shape: (N, 2)\n            offsets2d (torch.Tensor): The offsets between real centers2d\n                and base centers2d.\n                shape: (N , 2)\n            depths (torch.Tensor): Depths of objects.\n                shape: (N, )\n            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.\n                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)\n            downsample_ratio (int): The stride of feature map.\n            pad_mode (str, optional): Padding mode used in\n                training data augmentation.\n\n        Return:\n            tuple(torch.Tensor): Centers of 3D boxes.\n                shape: (N, 3)\n        "
        N = cam2imgs.shape[0]
        cam2imgs_inv = cam2imgs.inverse()
        if pad_mode == 'default':
            centers2d_img = (base_centers2d + offsets2d) * downsample_ratio
        else:
            raise NotImplementedError
        centers2d_img = torch.cat((centers2d_img, depths.unsqueeze(-1)), dim=1)
        centers2d_extend = torch.cat((centers2d_img, centers2d_img.new_ones(N, 1)), dim=1).unsqueeze(-1)
        locations = torch.matmul(cam2imgs_inv, centers2d_extend).squeeze(-1)
        return locations[:, :3]

    def keypoints2depth(self, keypoints2d, dimensions, cam2imgs, downsample_ratio=4, group0_index=[(7, 3), (0, 4)], group1_index=[(2, 6), (1, 5)]):
        if False:
            for i in range(10):
                print('nop')
        "Decode depth form three groups of keypoints and geometry projection\n        model. 2D keypoints inlucding 8 coreners and top/bottom centers will be\n        divided into three groups which will be used to calculate three depths\n        of object.\n\n        .. code-block:: none\n\n                Group center keypoints:\n\n                             + --------------- +\n                            /|   top center   /|\n                           / |      .        / |\n                          /  |      |       /  |\n                         + ---------|----- +   +\n                         |  /       |      |  /\n                         | /        .      | /\n                         |/ bottom center  |/\n                         + --------------- +\n\n                Group 0 keypoints:\n\n                             0\n                             + -------------- +\n                            /|               /|\n                           / |              / |\n                          /  |            5/  |\n                         + -------------- +   +\n                         |  /3            |  /\n                         | /              | /\n                         |/               |/\n                         + -------------- + 6\n\n                Group 1 keypoints:\n\n                                               4\n                             + -------------- +\n                            /|               /|\n                           / |              / |\n                          /  |             /  |\n                       1 + -------------- +   + 7\n                         |  /             |  /\n                         | /              | /\n                         |/               |/\n                       2 + -------------- +\n\n\n        Args:\n            keypoints2d (torch.Tensor): Keypoints of objects.\n                8 vertices + top/bottom center.\n                shape: (N, 10, 2)\n            dimensions (torch.Tensor): Dimensions of objetcts.\n                shape: (N, 3)\n            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.\n                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)\n            downsample_ratio (int, opitonal): The stride of feature map.\n                Defaults: 4.\n            group0_index(list[tuple[int]], optional): Keypoints group 0\n                of index to calculate the depth.\n                Defaults: [0, 3, 4, 7].\n            group1_index(list[tuple[int]], optional): Keypoints group 1\n                of index to calculate the depth.\n                Defaults: [1, 2, 5, 6]\n\n        Return:\n            tuple(torch.Tensor): Depth computed from three groups of\n                keypoints (top/bottom, group0, group1)\n                shape: (N, 3)\n        "
        pred_height_3d = dimensions[:, 1].clone()
        f_u = cam2imgs[:, 0, 0]
        center_height = keypoints2d[:, -2, 1] - keypoints2d[:, -1, 1]
        corner_group0_height = keypoints2d[:, group0_index[0], 1] - keypoints2d[:, group0_index[1], 1]
        corner_group1_height = keypoints2d[:, group1_index[0], 1] - keypoints2d[:, group1_index[1], 1]
        center_depth = f_u * pred_height_3d / (F.relu(center_height) * downsample_ratio + self.eps)
        corner_group0_depth = (f_u * pred_height_3d).unsqueeze(-1) / (F.relu(corner_group0_height) * downsample_ratio + self.eps)
        corner_group1_depth = (f_u * pred_height_3d).unsqueeze(-1) / (F.relu(corner_group1_height) * downsample_ratio + self.eps)
        corner_group0_depth = corner_group0_depth.mean(dim=1)
        corner_group1_depth = corner_group1_depth.mean(dim=1)
        keypoints_depth = torch.stack((center_depth, corner_group0_depth, corner_group1_depth), dim=1)
        keypoints_depth = torch.clamp(keypoints_depth, min=self.depth_range[0], max=self.depth_range[1])
        return keypoints_depth

    def decode_dims(self, labels, dims_offset):
        if False:
            for i in range(10):
                print('nop')
        "Retrieve object dimensions.\n\n        Args:\n            labels (torch.Tensor): Each points' category id.\n                shape: (N, K)\n            dims_offset (torch.Tensor): Dimension offsets.\n                shape: (N, 3)\n\n        Returns:\n            torch.Tensor: Shape (N, 3)\n        "
        if self.dims_mode == 'exp':
            dims_offset = dims_offset.exp()
        elif self.dims_mode == 'linear':
            labels = labels.long()
            base_dims = dims_offset.new_tensor(self.base_dims)
            dims_mean = base_dims[:, :3]
            dims_std = base_dims[:, 3:6]
            cls_dimension_mean = dims_mean[labels, :]
            cls_dimension_std = dims_std[labels, :]
            dimensions = dims_offset * cls_dimension_mean + cls_dimension_std
        else:
            raise ValueError
        return dimensions

    def decode_orientation(self, ori_vector, locations):
        if False:
            print('Hello World!')
        'Retrieve object orientation.\n\n        Args:\n            ori_vector (torch.Tensor): Local orientation vector\n                in [axis_cls, head_cls, sin, cos] format.\n                shape: (N, num_dir_bins * 4)\n            locations (torch.Tensor): Object location.\n                shape: (N, 3)\n\n        Returns:\n            tuple[torch.Tensor]: yaws and local yaws of 3d bboxes.\n        '
        if self.multibin:
            pred_bin_cls = ori_vector[:, :self.num_dir_bins * 2].view(-1, self.num_dir_bins, 2)
            pred_bin_cls = pred_bin_cls.softmax(dim=2)[..., 1]
            orientations = ori_vector.new_zeros(ori_vector.shape[0])
            for i in range(self.num_dir_bins):
                mask_i = pred_bin_cls.argmax(dim=1) == i
                start_bin = self.num_dir_bins * 2 + i * 2
                end_bin = start_bin + 2
                pred_bin_offset = ori_vector[mask_i, start_bin:end_bin]
                orientations[mask_i] = pred_bin_offset[:, 0].atan2(pred_bin_offset[:, 1]) + self.bin_centers[i]
        else:
            axis_cls = ori_vector[:, :2].softmax(dim=1)
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = ori_vector[:, 2:4].softmax(dim=1)
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            orientations = self.bin_centers[axis_cls + head_cls * 2]
            sin_cos_offset = F.normalize(ori_vector[:, 4:])
            orientations += sin_cos_offset[:, 0].atan(sin_cos_offset[:, 1])
        locations = locations.view(-1, 3)
        rays = locations[:, 0].atan2(locations[:, 2])
        local_yaws = orientations
        yaws = local_yaws + rays
        larger_idx = (yaws > np.pi).nonzero(as_tuple=False)
        small_idx = (yaws < -np.pi).nonzero(as_tuple=False)
        if len(larger_idx) != 0:
            yaws[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            yaws[small_idx] += 2 * np.pi
        larger_idx = (local_yaws > np.pi).nonzero(as_tuple=False)
        small_idx = (local_yaws < -np.pi).nonzero(as_tuple=False)
        if len(larger_idx) != 0:
            local_yaws[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            local_yaws[small_idx] += 2 * np.pi
        return (yaws, local_yaws)

    def decode_bboxes2d(self, reg_bboxes2d, base_centers2d):
        if False:
            print('Hello World!')
        'Retrieve [x1, y1, x2, y2] format 2D bboxes.\n\n        Args:\n            reg_bboxes2d (torch.Tensor): Predicted FCOS style\n                2D bboxes.\n                shape: (N, 4)\n            base_centers2d (torch.Tensor): predicted base centers2d.\n                shape: (N, 2)\n\n        Returns:\n            torch.Tenosr: [x1, y1, x2, y2] format 2D bboxes.\n        '
        centers_x = base_centers2d[:, 0]
        centers_y = base_centers2d[:, 1]
        xs_min = centers_x - reg_bboxes2d[..., 0]
        ys_min = centers_y - reg_bboxes2d[..., 1]
        xs_max = centers_x + reg_bboxes2d[..., 2]
        ys_max = centers_y + reg_bboxes2d[..., 3]
        bboxes2d = torch.stack([xs_min, ys_min, xs_max, ys_max], dim=-1)
        return bboxes2d

    def combine_depths(self, depth, depth_uncertainty):
        if False:
            while True:
                i = 10
        'Combine all the prediced depths with depth uncertainty.\n\n        Args:\n            depth (torch.Tensor): Predicted depths of each object.\n                2D bboxes.\n                shape: (N, 4)\n            depth_uncertainty (torch.Tensor): Depth uncertainty for\n                each depth of each object.\n                shape: (N, 4)\n\n        Returns:\n            torch.Tenosr: combined depth.\n        '
        uncertainty_weights = 1 / depth_uncertainty
        uncertainty_weights = uncertainty_weights / uncertainty_weights.sum(dim=1, keepdim=True)
        combined_depth = torch.sum(depth * uncertainty_weights, dim=1)
        return combined_depth
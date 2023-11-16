import warnings
from abc import abstractmethod
import numpy as np
import torch
from mmcv.ops import box_iou_rotated, points_in_boxes_all, points_in_boxes_part
from .utils import limit_period

class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Defaults to True.
        origin (tuple[float], optional): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if False:
            i = 10
            return i + 15
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()
        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        if False:
            i = 10
            return i + 15
        'torch.Tensor: A vector with volume of each box.'
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        if False:
            for i in range(10):
                print('nop')
        'torch.Tensor: Size dimensions of each box in shape (N, 3).'
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        if False:
            return 10
        'torch.Tensor: A vector with yaw of each box in shape (N, ).'
        return self.tensor[:, 6]

    @property
    def height(self):
        if False:
            while True:
                i = 10
        'torch.Tensor: A vector with height of each box in shape (N, ).'
        return self.tensor[:, 5]

    @property
    def top_height(self):
        if False:
            print('Hello World!')
        'torch.Tensor:\n            A vector with the top height of each box in shape (N, ).'
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        if False:
            return 10
        "torch.Tensor:\n            A vector with bottom's height of each box in shape (N, )."
        return self.tensor[:, 2]

    @property
    def center(self):
        if False:
            while True:
                i = 10
        "Calculate the center of all the boxes.\n\n        Note:\n            In MMDetection3D's convention, the bottom center is\n            usually taken as the default center.\n\n            The relative position of the centers in different kinds of\n            boxes are different, e.g., the relative center of a boxes is\n            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.\n            It is recommended to use ``bottom_center`` or ``gravity_center``\n            for clearer usage.\n\n        Returns:\n            torch.Tensor: A tensor with center of each box in shape (N, 3).\n        "
        return self.bottom_center

    @property
    def bottom_center(self):
        if False:
            for i in range(10):
                print('nop')
        'torch.Tensor: A tensor with center of each box in shape (N, 3).'
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        if False:
            print('Hello World!')
        'torch.Tensor: A tensor with center of each box in shape (N, 3).'
        pass

    @property
    def corners(self):
        if False:
            print('Hello World!')
        'torch.Tensor:\n            a tensor with 8 corners of each box in shape (N, 8, 3).'
        pass

    @property
    def bev(self):
        if False:
            return 10
        'torch.Tensor: 2D BEV box of each box with rotation\n            in XYWHR format, in shape (N, 5).'
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        if False:
            print('Hello World!')
        'torch.Tensor: A tensor of 2D BEV box of each box\n            without rotation.'
        bev_rotated_boxes = self.bev
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4])
        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def in_range_bev(self, box_range):
        if False:
            while True:
                i = 10
        'Check whether the boxes are in the given range.\n\n        Args:\n            box_range (list | torch.Tensor): the range of box\n                (x_min, y_min, x_max, y_max)\n\n        Note:\n            The original implementation of SECOND checks whether boxes in\n            a range by checking whether the points are in a convex\n            polygon, we reduce the burden for simpler cases.\n\n        Returns:\n            torch.Tensor: Whether each box is inside the reference range.\n        '
        in_range_flags = (self.bev[:, 0] > box_range[0]) & (self.bev[:, 1] > box_range[1]) & (self.bev[:, 0] < box_range[2]) & (self.bev[:, 1] < box_range[3])
        return in_range_flags

    @abstractmethod
    def rotate(self, angle, points=None):
        if False:
            for i in range(10):
                print('nop')
        'Rotate boxes with points (optional) with the given angle or rotation\n        matrix.\n\n        Args:\n            angle (float | torch.Tensor | np.ndarray):\n                Rotation angle or rotation matrix.\n            points (torch.Tensor | numpy.ndarray |\n                :obj:`BasePoints`, optional):\n                Points to rotate. Defaults to None.\n        '
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        if False:
            print('Hello World!')
        "Flip the boxes in BEV along given BEV direction.\n\n        Args:\n            bev_direction (str, optional): Direction by which to flip.\n                Can be chosen from 'horizontal' and 'vertical'.\n                Defaults to 'horizontal'.\n        "
        pass

    def translate(self, trans_vector):
        if False:
            while True:
                i = 10
        'Translate boxes with the given translation vector.\n\n        Args:\n            trans_vector (torch.Tensor): Translation vector of size (1, 3).\n        '
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        if False:
            i = 10
            return i + 15
        'Check whether the boxes are in the given range.\n\n        Args:\n            box_range (list | torch.Tensor): The range of box\n                (x_min, y_min, z_min, x_max, y_max, z_max)\n\n        Note:\n            In the original implementation of SECOND, checking whether\n            a box in the range checks whether the points are in a convex\n            polygon, we try to reduce the burden for simpler cases.\n\n        Returns:\n            torch.Tensor: A binary vector indicating whether each box is\n                inside the reference range.\n        '
        in_range_flags = (self.tensor[:, 0] > box_range[0]) & (self.tensor[:, 1] > box_range[1]) & (self.tensor[:, 2] > box_range[2]) & (self.tensor[:, 0] < box_range[3]) & (self.tensor[:, 1] < box_range[4]) & (self.tensor[:, 2] < box_range[5])
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        if False:
            return 10
        'Convert self to ``dst`` mode.\n\n        Args:\n            dst (:obj:`Box3DMode`): The target Box mode.\n            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and\n                translation matrix between different coordinates.\n                Defaults to None.\n                The conversion from `src` coordinates to `dst` coordinates\n                usually comes along the change of sensors, e.g., from camera\n                to LiDAR. This requires a transformation matrix.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: The converted box of the same type\n                in the `dst` mode.\n        '
        pass

    def scale(self, scale_factor):
        if False:
            while True:
                i = 10
        'Scale the box with horizontal and vertical scaling factors.\n\n        Args:\n            scale_factors (float): Scale factors to scale the boxes.\n        '
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        if False:
            return 10
        'Limit the yaw to a given period and offset.\n\n        Args:\n            offset (float, optional): The offset of the yaw. Defaults to 0.5.\n            period (float, optional): The expected period. Defaults to np.pi.\n        '
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold=0.0):
        if False:
            i = 10
            return i + 15
        'Find boxes that are non-empty.\n\n        A box is considered empty,\n        if either of its side is no larger than threshold.\n\n        Args:\n            threshold (float, optional): The threshold of minimal sizes.\n                Defaults to 0.0.\n\n        Returns:\n            torch.Tensor: A binary vector which represents whether each\n                box is empty (False) or non-empty (True).\n        '
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        if False:
            return 10
        "\n        Note:\n            The following usage are allowed:\n            1. `new_boxes = boxes[3]`:\n                return a `Boxes` that contains only one box.\n            2. `new_boxes = boxes[2:10]`:\n                return a slice of boxes.\n            3. `new_boxes = boxes[vector]`:\n                where vector is a torch.BoolTensor with `length = len(boxes)`.\n                Nonzero elements in the vector will be selected.\n            Note that the returned Boxes might share storage with this Boxes,\n            subject to Pytorch's indexing semantics.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: A new object of\n                :class:`BaseInstance3DBoxes` after indexing.\n        "
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        if False:
            while True:
                i = 10
        'int: Number of boxes in the current object.'
        return self.tensor.shape[0]

    def __repr__(self):
        if False:
            while True:
                i = 10
        'str: Return a strings that describes the object.'
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list):
        if False:
            print('Hello World!')
        'Concatenate a list of Boxes into a single Boxes.\n\n        Args:\n            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.\n        '
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all((isinstance(box, cls) for box in boxes_list))
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0), box_dim=boxes_list[0].tensor.shape[1], with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def to(self, device):
        if False:
            return 10
        'Convert current boxes to a specific device.\n\n        Args:\n            device (str | :obj:`torch.device`): The name of the device.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: A new boxes object on the\n                specific device.\n        '
        original_type = type(self)
        return original_type(self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def clone(self):
        if False:
            return 10
        'Clone the Boxes.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: Box object with the same properties\n                as self.\n        '
        original_type = type(self)
        return original_type(self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self):
        if False:
            while True:
                i = 10
        'str: The device of the boxes are on.'
        return self.tensor.device

    def __iter__(self):
        if False:
            return 10
        'Yield a box as a Tensor of shape (4,) at a time.\n\n        Returns:\n            torch.Tensor: A box of shape (4,).\n        '
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        if False:
            for i in range(10):
                print('nop')
        "Calculate height overlaps of two boxes.\n\n        Note:\n            This function calculates the height overlaps between boxes1 and\n            boxes2,  boxes1 and boxes2 should be in the same type.\n\n        Args:\n            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.\n            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.\n            mode (str, optional): Mode of IoU calculation. Defaults to 'iou'.\n\n        Returns:\n            torch.Tensor: Calculated iou of boxes.\n        "
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" shouldbe in the same type, got {type(boxes1)} and {type(boxes2)}.'
        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)
        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        if False:
            while True:
                i = 10
        "Calculate 3D overlaps of two boxes.\n\n        Note:\n            This function calculates the overlaps between ``boxes1`` and\n            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.\n\n        Args:\n            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.\n            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.\n            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.\n\n        Returns:\n            torch.Tensor: Calculated 3D overlaps of the boxes.\n        "
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" shouldbe in the same type, got {type(boxes1)} and {type(boxes2)}.'
        assert mode in ['iou', 'iof']
        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)
        overlaps_h = cls.height_overlaps(boxes1, boxes2)
        iou2d = box_iou_rotated(boxes1.bev, boxes2.bev)
        areas1 = (boxes1.bev[:, 2] * boxes1.bev[:, 3]).unsqueeze(1).expand(rows, cols)
        areas2 = (boxes2.bev[:, 2] * boxes2.bev[:, 3]).unsqueeze(0).expand(rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h
        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)
        if mode == 'iou':
            iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-08)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-08)
        return iou3d

    def new_box(self, data):
        if False:
            for i in range(10):
                print('nop')
        "Create a new box object with data.\n\n        The new box and its tensor has the similar properties\n            as self and self.tensor, respectively.\n\n        Args:\n            data (torch.Tensor | numpy.array | list): Data to be copied.\n\n        Returns:\n            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,\n                the object's other properties are similar to ``self``.\n        "
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def points_in_boxes_part(self, points, boxes_override=None):
        if False:
            while True:
                i = 10
        'Find the box in which each point is.\n\n        Args:\n            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),\n                3 dimensions are (x, y, z) in LiDAR or depth coordinate.\n            boxes_override (torch.Tensor, optional): Boxes to override\n                `self.tensor`. Defaults to None.\n\n        Returns:\n            torch.Tensor: The index of the first box that each point\n                is in, in shape (M, ). Default value is -1\n                (if the point is not enclosed by any box).\n\n        Note:\n            If a point is enclosed by multiple boxes, the index of the\n            first box will be returned.\n        '
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        if points.dim() == 2:
            points = points.unsqueeze(0)
        box_idx = points_in_boxes_part(points, boxes.unsqueeze(0).to(points.device)).squeeze(0)
        return box_idx

    def points_in_boxes_all(self, points, boxes_override=None):
        if False:
            return 10
        'Find all boxes in which each point is.\n\n        Args:\n            points (torch.Tensor): Points in shape (1, M, 3) or (M, 3),\n                3 dimensions are (x, y, z) in LiDAR or depth coordinate.\n            boxes_override (torch.Tensor, optional): Boxes to override\n                `self.tensor`. Defaults to None.\n\n        Returns:\n            torch.Tensor: A tensor indicating whether a point is in a box,\n                in shape (M, T). T is the number of boxes. Denote this\n                tensor as A, if the m^th point is in the t^th box, then\n                `A[m, t] == 1`, elsewise `A[m, t] == 0`.\n        '
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1
        boxes = boxes.to(points_clone.device).unsqueeze(0)
        box_idxs_of_pts = points_in_boxes_all(points_clone, boxes)
        return box_idxs_of_pts.squeeze(0)

    def points_in_boxes(self, points, boxes_override=None):
        if False:
            while True:
                i = 10
        warnings.warn('DeprecationWarning: points_in_boxes is a deprecated method, please consider using points_in_boxes_part.')
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(self, points, boxes_override=None):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('DeprecationWarning: points_in_boxes_batch is a deprecated method, please consider using points_in_boxes_all.')
        return self.points_in_boxes_all(points, boxes_override)
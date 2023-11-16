import warnings
from abc import abstractmethod
import numpy as np
import torch
from ..bbox.structures.utils import rotation_3d_in_axis

class BasePoints(object):
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int, optional): Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the
            meaning of extra dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if False:
            return 10
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, points_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()
        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self):
        if False:
            while True:
                i = 10
        'torch.Tensor: Coordinates of each point in shape (N, 3).'
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        if False:
            i = 10
            return i + 15
        'Set the coordinates of each point.'
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        if False:
            return 10
        'torch.Tensor:\n            A vector with height of each point in shape (N, 1), or None.'
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor):
        if False:
            return 10
        'Set the height of each point.'
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        'torch.Tensor:\n            A vector with color of each point in shape (N, 3), or None.'
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor):
        if False:
            print('Hello World!')
        'Set the color of each point.'
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self):
        if False:
            return 10
        'torch.Shape: Shape of points.'
        return self.tensor.shape

    def shuffle(self):
        if False:
            print('Hello World!')
        'Shuffle the points.\n\n        Returns:\n            torch.Tensor: The shuffled index.\n        '
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        if False:
            return 10
        'Rotate points with the given rotation matrix or angle.\n\n        Args:\n            rotation (float | np.ndarray | torch.Tensor): Rotation matrix\n                or angle.\n            axis (int, optional): Axis to rotate at. Defaults to None.\n        '
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'
        if axis is None:
            axis = self.rotation_axis
        if rotation.numel() == 1:
            (rotated_points, rot_mat_T) = rotation_3d_in_axis(self.tensor[:, :3][None], rotation, axis=axis, return_mat=True)
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation
        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        if False:
            i = 10
            return i + 15
        'Flip the points along given BEV direction.\n\n        Args:\n            bev_direction (str): Flip direction (horizontal or vertical).\n        '
        pass

    def translate(self, trans_vector):
        if False:
            i = 10
            return i + 15
        'Translate points with the given translation vector.\n\n        Args:\n            trans_vector (np.ndarray, torch.Tensor): Translation\n                vector of size 3 or nx3.\n        '
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f'Unsupported translation vector of shape {trans_vector.shape}')
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the points are in the given range.\n\n        Args:\n            point_range (list | torch.Tensor): The range of point\n                (x_min, y_min, z_min, x_max, y_max, z_max)\n\n        Note:\n            In the original implementation of SECOND, checking whether\n            a box in the range checks whether the points are in a convex\n            polygon, we try to reduce the burden for simpler cases.\n\n        Returns:\n            torch.Tensor: A binary vector indicating whether each point is\n                inside the reference range.\n        '
        in_range_flags = (self.tensor[:, 0] > point_range[0]) & (self.tensor[:, 1] > point_range[1]) & (self.tensor[:, 2] > point_range[2]) & (self.tensor[:, 0] < point_range[3]) & (self.tensor[:, 1] < point_range[4]) & (self.tensor[:, 2] < point_range[5])
        return in_range_flags

    @property
    def bev(self):
        if False:
            i = 10
            return i + 15
        'torch.Tensor: BEV of the points in shape (N, 2).'
        return self.tensor[:, [0, 1]]

    def in_range_bev(self, point_range):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the points are in the given range.\n\n        Args:\n            point_range (list | torch.Tensor): The range of point\n                in order of (x_min, y_min, x_max, y_max).\n\n        Returns:\n            torch.Tensor: Indicating whether each point is inside\n                the reference range.\n        '
        in_range_flags = (self.bev[:, 0] > point_range[0]) & (self.bev[:, 1] > point_range[1]) & (self.bev[:, 0] < point_range[2]) & (self.bev[:, 1] < point_range[3])
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        if False:
            while True:
                i = 10
        'Convert self to ``dst`` mode.\n\n        Args:\n            dst (:obj:`CoordMode`): The target Box mode.\n            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and\n                translation matrix between different coordinates.\n                Defaults to None.\n                The conversion from `src` coordinates to `dst` coordinates\n                usually comes along the change of sensors, e.g., from camera\n                to LiDAR. This requires a transformation matrix.\n\n        Returns:\n            :obj:`BasePoints`: The converted box of the same type\n                in the `dst` mode.\n        '
        pass

    def scale(self, scale_factor):
        if False:
            i = 10
            return i + 15
        'Scale the points with horizontal and vertical scaling factors.\n\n        Args:\n            scale_factors (float): Scale factors to scale the points.\n        '
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        "\n        Note:\n            The following usage are allowed:\n            1. `new_points = points[3]`:\n                return a `Points` that contains only one point.\n            2. `new_points = points[2:10]`:\n                return a slice of points.\n            3. `new_points = points[vector]`:\n                where vector is a torch.BoolTensor with `length = len(points)`.\n                Nonzero elements in the vector will be selected.\n            4. `new_points = points[3:11, vector]`:\n                return a slice of points and attribute dims.\n            5. `new_points = points[4:12, 2]`:\n                return a slice of points with single attribute.\n            Note that the returned Points might share storage with this Points,\n            subject to Pytorch's indexing semantics.\n\n        Returns:\n            :obj:`BasePoints`: A new object of\n                :class:`BasePoints` after indexing.\n        "
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), points_dim=self.points_dim, attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]
            keep_dims = list(set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')
        assert p.dim() == 2, f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self):
        if False:
            print('Hello World!')
        'int: Number of points in the current object.'
        return self.tensor.shape[0]

    def __repr__(self):
        if False:
            print('Hello World!')
        'str: Return a strings that describes the object.'
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list):
        if False:
            return 10
        'Concatenate a list of Points into a single Points.\n\n        Args:\n            points_list (list[:obj:`BasePoints`]): List of points.\n\n        Returns:\n            :obj:`BasePoints`: The concatenated Points.\n        '
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all((isinstance(points, cls) for points in points_list))
        cat_points = cls(torch.cat([p.tensor for p in points_list], dim=0), points_dim=points_list[0].tensor.shape[1], attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def to(self, device):
        if False:
            while True:
                i = 10
        'Convert current points to a specific device.\n\n        Args:\n            device (str | :obj:`torch.device`): The name of the device.\n\n        Returns:\n            :obj:`BasePoints`: A new boxes object on the\n                specific device.\n        '
        original_type = type(self)
        return original_type(self.tensor.to(device), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def clone(self):
        if False:
            i = 10
            return i + 15
        'Clone the Points.\n\n        Returns:\n            :obj:`BasePoints`: Box object with the same properties\n                as self.\n        '
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self):
        if False:
            print('Hello World!')
        'str: The device of the points are on.'
        return self.tensor.device

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Yield a point as a Tensor of shape (4,) at a time.\n\n        Returns:\n            torch.Tensor: A point of shape (4,).\n        '
        yield from self.tensor

    def new_point(self, data):
        if False:
            print('Hello World!')
        "Create a new point object with data.\n\n        The new point and its tensor has the similar properties\n            as self and self.tensor, respectively.\n\n        Args:\n            data (torch.Tensor | numpy.array | list): Data to be copied.\n\n        Returns:\n            :obj:`BasePoints`: A new point object with ``data``,\n                the object's other properties are similar to ``self``.\n        "
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)
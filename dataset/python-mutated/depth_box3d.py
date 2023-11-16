import numpy as np
import torch
from mmdet3d.core.points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis

class DepthInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in Depth coordinates.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (yaw=-0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (yaw=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        if False:
            i = 10
            return i + 15
        'torch.Tensor: A tensor with center of each box in shape (N, 3).'
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        if False:
            i = 10
            return i + 15
        'torch.Tensor: Coordinates of corners of all the boxes\n        in shape (N, 8, 3).\n\n        Convert the boxes to corners in clockwise order, in form of\n        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``\n\n        .. code-block:: none\n\n                                           up z\n                            front y           ^\n                                 /            |\n                                /             |\n                  (x0, y1, z1) + -----------  + (x1, y1, z1)\n                              /|            / |\n                             / |           /  |\n               (x0, y0, z1) + ----------- +   + (x1, y1, z0)\n                            |  /      .   |  /\n                            | / origin    | /\n               (x0, y0, z0) + ----------- + --------> right x\n                                          (x1, y0, z0)\n        '
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)
        dims = self.dims
        corners_norm = torch.from_numpy(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(device=dims.device, dtype=dims.dtype)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(self, angle, points=None):
        if False:
            while True:
                i = 10
        'Rotate boxes with points (optional) with the given angle or rotation\n        matrix.\n\n        Args:\n            angle (float | torch.Tensor | np.ndarray):\n                Rotation angle or rotation matrix.\n            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):\n                Points to rotate. Defaults to None.\n\n        Returns:\n            tuple or None: When ``points`` is None, the function returns\n                None, otherwise it returns the rotated points and the\n                rotation matrix ``rot_mat_T``.\n        '
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, f'invalid rotation angle shape {angle.shape}'
        if angle.numel() == 1:
            (self.tensor[:, 0:3], rot_mat_T) = rotation_3d_in_axis(self.tensor[:, 0:3], angle, axis=self.YAW_AXIS, return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T
        if self.with_yaw:
            self.tensor[:, 6] += angle
        else:
            corners_rot = self.corners @ rot_mat_T
            new_x_size = corners_rot[..., 0].max(dim=1, keepdim=True)[0] - corners_rot[..., 0].min(dim=1, keepdim=True)[0]
            new_y_size = corners_rot[..., 1].max(dim=1, keepdim=True)[0] - corners_rot[..., 1].min(dim=1, keepdim=True)[0]
            self.tensor[:, 3:5] = torch.cat((new_x_size, new_y_size), dim=-1)
        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return (points, rot_mat_T)

    def flip(self, bev_direction='horizontal', points=None):
        if False:
            while True:
                i = 10
        "Flip the boxes in BEV along given BEV direction.\n\n        In Depth coordinates, it flips x (horizontal) or y (vertical) axis.\n\n        Args:\n            bev_direction (str, optional): Flip direction\n                (horizontal or vertical). Defaults to 'horizontal'.\n            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):\n                Points to flip. Defaults to None.\n\n        Returns:\n            torch.Tensor, numpy.ndarray or None: Flipped points.\n        "
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self, dst, rt_mat=None):
        if False:
            return 10
        'Convert self to ``dst`` mode.\n\n        Args:\n            dst (:obj:`Box3DMode`): The target Box mode.\n            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and\n                translation matrix between different coordinates.\n                Defaults to None.\n                The conversion from ``src`` coordinates to ``dst`` coordinates\n                usually comes along the change of sensors, e.g., from camera\n                to LiDAR. This requires a transformation matrix.\n\n        Returns:\n            :obj:`DepthInstance3DBoxes`:\n                The converted box of the same type in the ``dst`` mode.\n        '
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(box=self, src=Box3DMode.DEPTH, dst=dst, rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        if False:
            i = 10
            return i + 15
        'Enlarge the length, width and height boxes.\n\n        Args:\n            extra_width (float | torch.Tensor): Extra width to enlarge the box.\n\n        Returns:\n            :obj:`DepthInstance3DBoxes`: Enlarged boxes.\n        '
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def get_surface_line_center(self):
        if False:
            return 10
        'Compute surface and line center of bounding boxes.\n\n        Returns:\n            torch.Tensor: Surface and line center of bounding boxes.\n        '
        obj_size = self.dims
        center = self.gravity_center.view(-1, 1, 3)
        batch_size = center.shape[0]
        rot_sin = torch.sin(-self.yaw)
        rot_cos = torch.cos(-self.yaw)
        rot_mat_T = self.yaw.new_zeros(tuple(list(self.yaw.shape) + [3, 3]))
        rot_mat_T[..., 0, 0] = rot_cos
        rot_mat_T[..., 0, 1] = -rot_sin
        rot_mat_T[..., 1, 0] = rot_sin
        rot_mat_T[..., 1, 1] = rot_cos
        rot_mat_T[..., 2, 2] = 1
        offset = obj_size.new_tensor([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        offset = offset.view(1, 6, 3) / 2
        surface_3d = (offset * obj_size.view(batch_size, 1, 3).repeat(1, 6, 1)).reshape(-1, 3)
        offset = obj_size.new_tensor([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1], [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
        offset = offset.view(1, 12, 3) / 2
        line_3d = (offset * obj_size.view(batch_size, 1, 3).repeat(1, 12, 1)).reshape(-1, 3)
        surface_rot = rot_mat_T.repeat(6, 1, 1)
        surface_3d = torch.matmul(surface_3d.unsqueeze(-2), surface_rot).squeeze(-2)
        surface_center = center.repeat(1, 6, 1).reshape(-1, 3) + surface_3d
        line_rot = rot_mat_T.repeat(12, 1, 1)
        line_3d = torch.matmul(line_3d.unsqueeze(-2), line_rot).squeeze(-2)
        line_center = center.repeat(1, 12, 1).reshape(-1, 3) + line_3d
        return (surface_center, line_center)
import functools
import torch
import torch.nn.functional as F

def quaternion_to_matrix(quaternions):
    if False:
        while True:
            i = 10
    '\n    Convert rotations given as quaternions to rotation matrices.\n\n    Args:\n        quaternions: quaternions with real part first,\n            as tensor of shape (..., 4).\n\n    Returns:\n        Rotation matrices as tensor of shape (..., 3, 3).\n    '
    (r, i, j, k) = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r), two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r), two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _axis_angle_rotation(axis: str, angle):
    if False:
        i = 10
        return i + 15
    '\n    Return the rotation matrices for one of the rotations about an axis\n    of which Euler angles describe, for each value of the angle given.\n\n    Args:\n        axis: Axis label "X" or "Y or "Z".\n        angle: any shape tensor of Euler angles in radians\n\n    Returns:\n        Rotation matrices as tensor of shape (..., 3, 3).\n    '
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == 'X':
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == 'Y':
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == 'Z':
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention: str):
    if False:
        while True:
            i = 10
    '\n    Convert rotations given as Euler angles in radians to rotation matrices.\n\n    Args:\n        euler_angles: Euler angles in radians as tensor of shape (..., 3).\n        convention: Convention string of three uppercase letters from\n            {"X", "Y", and "Z"}.\n\n    Returns:\n        Rotation matrices as tensor of shape (..., 3, 3).\n    '
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError('Invalid input euler angles.')
    if len(convention) != 3:
        raise ValueError('Convention must have 3 letters.')
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f'Invalid convention {convention}.')
    for letter in convention:
        if letter not in ('X', 'Y', 'Z'):
            raise ValueError(f'Invalid letter {letter} in convention string.')
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

def axis_angle_to_matrix(axis_angle):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert rotations given as axis/angle to rotation matrices.\n\n    Args:\n        axis_angle: Rotations given as a vector in axis angle form,\n            as a tensor of shape (..., 3), where the magnitude is\n            the angle turned anticlockwise in radians around the\n            vector's direction.\n\n    Returns:\n        Rotation matrices as tensor of shape (..., 3, 3).\n    "
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    '\n    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix\n    using Gram--Schmidt orthogonalisation per Section B of [1].\n    Args:\n        d6: 6D rotation representation, of size (*, 6)\n\n    Returns:\n        batch of rotation matrices of size (*, 3, 3)\n\n    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.\n    On the Continuity of Rotation Representations in Neural Networks.\n    IEEE Conference on Computer Vision and Pattern Recognition, 2019.\n    Retrieved from http://arxiv.org/abs/1812.07035\n    '
    (a1, a2) = (d6[..., :3], d6[..., 3:])
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
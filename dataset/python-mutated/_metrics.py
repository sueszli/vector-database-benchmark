"""Module including useful metrics for Structure from Motion."""
from torch import Tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import point_line_distance

def sampson_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool=True, eps: float=1e-08) -> Tensor:
    if False:
        while True:
            i = 10
    'Return Sampson distance for correspondences given the fundamental matrix.\n\n    Args:\n        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,\n              converted automatically.\n        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,\n              converted automatically.\n        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.\n        squared: if True (default), the squared distance is returned.\n        eps: Small constant for safe sqrt.\n\n    Returns:\n        the computed Sampson distance with shape :math:`(*, N)`.\n    '
    if not isinstance(Fm, Tensor):
        raise TypeError(f'Fm type is not a torch.Tensor. Got {type(Fm)}')
    if len(Fm.shape) < 3 or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f'Fm must be a (*, 3, 3) tensor. Got {Fm.shape}')
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)
    denominator: Tensor = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    out: Tensor = numerator / denominator
    if squared:
        return out
    return (out + eps).sqrt()

def symmetrical_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor, squared: bool=True, eps: float=1e-08) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Return symmetrical epipolar distance for correspondences given the fundamental matrix.\n\n    Args:\n       pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,\n             converted automatically.\n       pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,\n             converted automatically.\n       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.\n       squared: if True (default), the squared distance is returned.\n       eps: Small constant for safe sqrt.\n\n    Returns:\n        the computed Symmetrical distance with shape :math:`(*, N)`.\n    '
    if not isinstance(Fm, Tensor):
        raise TypeError(f'Fm type is not a torch.Tensor. Got {type(Fm)}')
    if len(Fm.shape) < 3 or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f'Fm must be a (*, 3, 3) tensor. Got {Fm.shape}')
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    line2_in_1: Tensor = pts2 @ Fm
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)
    denominator_inv: Tensor = 1.0 / line1_in_2[..., :2].norm(2, dim=-1).pow(2) + 1.0 / line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    out: Tensor = numerator * denominator_inv
    if squared:
        return out
    return (out + eps).sqrt()

def left_to_right_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Return one-sided epipolar distance for correspondences given the fundamental matrix.\n\n    This method measures the distance from points in the right images to the epilines\n    of the corresponding points in the left images as they reflect in the right images.\n\n    Args:\n       pts1: correspondences from the left images with shape\n         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.\n       pts2: correspondences from the right images with shape\n         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.\n       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to\n         avoid ambiguity with torch.nn.functional.\n\n    Returns:\n        the computed Symmetrical distance with shape :math:`(*, N)`.\n    '
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)
    if len(Fm.shape) < 3 or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f'Fm must be a (*, 3, 3) tensor. Got {Fm.shape}')
    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)
    F_t: Tensor = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2: Tensor = pts1 @ F_t
    return point_line_distance(pts2, line1_in_2)

def right_to_left_epipolar_distance(pts1: Tensor, pts2: Tensor, Fm: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    'Return one-sided epipolar distance for correspondences given the fundamental matrix.\n\n    This method measures the distance from points in the left images to the epilines\n    of the corresponding points in the right images as they reflect in the left images.\n\n    Args:\n       pts1: correspondences from the left images with shape\n         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.\n       pts2: correspondences from the right images with shape\n         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.\n       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to\n         avoid ambiguity with torch.nn.functional.\n\n    Returns:\n        the computed Symmetrical distance with shape :math:`(*, N)`.\n    '
    KORNIA_CHECK_IS_TENSOR(pts1)
    KORNIA_CHECK_IS_TENSOR(pts2)
    KORNIA_CHECK_IS_TENSOR(Fm)
    if len(Fm.shape) < 3 or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f'Fm must be a (*, 3, 3) tensor. Got {Fm.shape}')
    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)
    line2_in_1: Tensor = pts2 @ Fm
    return point_line_distance(pts1, line2_in_1)
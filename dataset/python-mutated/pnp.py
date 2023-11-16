from typing import Optional, Tuple
import torch
from torch.linalg import qr as linalg_qr
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points
from kornia.utils import eye_like
from kornia.utils.helpers import _torch_linalg_svdvals

def _mean_isotropic_scale_normalize(points: torch.Tensor, eps: float=1e-08) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        while True:
            i = 10
    'Normalizes points.\n\n    Args:\n       points : Tensor containing the points to be normalized with shape :math:`(B, N, D)`.\n       eps : Small value to avoid division by zero error.\n\n    Returns:\n       Tuple containing the normalized points in the shape :math:`(B, N, D)` and the transformation matrix\n       in the shape :math:`(B, D+1, D+1)`.\n    '
    if not isinstance(points, torch.Tensor):
        raise AssertionError(f'points is not an instance of torch.Tensor. Type of points is {type(points)}')
    if len(points.shape) != 3:
        raise AssertionError(f'points must be of shape (B, N, D). Got shape {points.shape}.')
    x_mean = torch.mean(points, dim=1, keepdim=True)
    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)
    D_int = points.shape[-1]
    D_float = torch.tensor(points.shape[-1], dtype=torch.float64, device=points.device)
    scale = torch.sqrt(D_float) / (scale + eps)
    transform = eye_like(D_int + 1, points)
    idxs = torch.arange(D_int, dtype=torch.int64, device=points.device)
    transform[:, idxs, idxs] = transform[:, idxs, idxs] * scale[:, None]
    transform[:, idxs, D_int] = transform[:, idxs, D_int] + -scale[:, None] * x_mean[:, 0, idxs]
    points_norm = transform_points(transform, points)
    return (points_norm, transform)

def solve_pnp_dlt(world_points: torch.Tensor, img_points: torch.Tensor, intrinsics: torch.Tensor, weights: Optional[torch.Tensor]=None, svd_eps: float=0.0001) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    'This function attempts to solve the Perspective-n-Point (PnP) problem using Direct Linear Transform (DLT).\n\n    Given a batch (where batch size is :math:`B`) of :math:`N` 3D points\n    (where :math:`N \\geq 6`) in the world space, a batch of :math:`N`\n    corresponding 2D points in the image space and a batch of\n    intrinsic matrices, this function tries to estimate a batch of\n    world to camera transformation matrices.\n\n    This implementation needs at least 6 points (i.e. :math:`N \\geq 6`) to\n    provide solutions.\n\n    This function cannot be used if all the 3D world points (of any element\n    of the batch) lie on a line or if all the 3D world points (of any element\n    of the batch) lie on a plane. This function attempts to check for these\n    conditions and throws an AssertionError if found. Do note that this check\n    is sensitive to the value of the svd_eps parameter.\n\n    Another bad condition occurs when the camera and the points lie on a\n    twisted cubic. However, this function does not check for this condition.\n\n    Args:\n        world_points : A tensor with shape :math:`(B, N, 3)` representing\n          the points in the world space.\n        img_points : A tensor with shape :math:`(B, N, 2)` representing\n          the points in the image space.\n        intrinsics : A tensor with shape :math:`(B, 3, 3)` representing\n          the intrinsic matrices.\n        weights : This parameter is not used currently and is just a\n          placeholder for API consistency.\n        svd_eps : A small float value to avoid numerical precision issues.\n\n    Returns:\n        A tensor with shape :math:`(B, 3, 4)` representing the estimated world to\n        camera transformation matrices (also known as the extrinsic matrices).\n\n    Example:\n        >>> world_points = torch.tensor([[\n        ...     [ 5. , -5. ,  0. ], [ 0. ,  0. ,  1.5],\n        ...     [ 2.5,  3. ,  6. ], [ 9. , -2. ,  3. ],\n        ...     [-4. ,  5. ,  2. ], [-5. ,  5. ,  1. ],\n        ... ]], dtype=torch.float64)\n        >>>\n        >>> img_points = torch.tensor([[\n        ...     [1409.1504, -800.936 ], [ 407.0207, -182.1229],\n        ...     [ 392.7021,  177.9428], [1016.838 ,   -2.9416],\n        ...     [ -63.1116,  142.9204], [-219.3874,   99.666 ],\n        ... ]], dtype=torch.float64)\n        >>>\n        >>> intrinsics = torch.tensor([[\n        ...     [ 500.,    0.,  250.],\n        ...     [   0.,  500.,  250.],\n        ...     [   0.,    0.,    1.],\n        ... ]], dtype=torch.float64)\n        >>>\n        >>> print(world_points.shape, img_points.shape, intrinsics.shape)\n        torch.Size([1, 6, 3]) torch.Size([1, 6, 2]) torch.Size([1, 3, 3])\n        >>>\n        >>> pred_world_to_cam = kornia.geometry.solve_pnp_dlt(world_points, img_points, intrinsics)\n        >>>\n        >>> print(pred_world_to_cam.shape)\n        torch.Size([1, 3, 4])\n        >>>\n        >>> pred_world_to_cam\n        tensor([[[ 0.9392, -0.3432, -0.0130,  1.6734],\n                 [ 0.3390,  0.9324, -0.1254, -4.3634],\n                 [ 0.0552,  0.1134,  0.9920,  3.7785]]], dtype=torch.float64)\n    '
    if not isinstance(world_points, torch.Tensor):
        raise AssertionError(f'world_points is not an instance of torch.Tensor. Type of world_points is {type(world_points)}')
    if not isinstance(img_points, torch.Tensor):
        raise AssertionError(f'img_points is not an instance of torch.Tensor. Type of img_points is {type(img_points)}')
    if not isinstance(intrinsics, torch.Tensor):
        raise AssertionError(f'intrinsics is not an instance of torch.Tensor. Type of intrinsics is {type(intrinsics)}')
    if weights is not None and (not isinstance(weights, torch.Tensor)):
        raise AssertionError(f'If weights is not None, then weights should be an instance of torch.Tensor. Type of weights is {type(weights)}')
    if not isinstance(svd_eps, float):
        raise AssertionError(f'Type of svd_eps is not float. Got {type(svd_eps)}')
    accepted_dtypes = (torch.float32, torch.float64)
    if world_points.dtype not in accepted_dtypes:
        raise AssertionError(f'world_points must have one of the following dtypes {accepted_dtypes}. Currently it has {world_points.dtype}.')
    if img_points.dtype not in accepted_dtypes:
        raise AssertionError(f'img_points must have one of the following dtypes {accepted_dtypes}. Currently it has {img_points.dtype}.')
    if intrinsics.dtype not in accepted_dtypes:
        raise AssertionError(f'intrinsics must have one of the following dtypes {accepted_dtypes}. Currently it has {intrinsics.dtype}.')
    if len(world_points.shape) != 3 or world_points.shape[2] != 3:
        raise AssertionError(f'world_points must be of shape (B, N, 3). Got shape {world_points.shape}.')
    if len(img_points.shape) != 3 or img_points.shape[2] != 2:
        raise AssertionError(f'img_points must be of shape (B, N, 2). Got shape {img_points.shape}.')
    if len(intrinsics.shape) != 3 or intrinsics.shape[1:] != (3, 3):
        raise AssertionError(f'intrinsics must be of shape (B, 3, 3). Got shape {intrinsics.shape}.')
    if world_points.shape[1] != img_points.shape[1]:
        raise AssertionError('world_points and img_points must have equal number of points.')
    if world_points.shape[0] != img_points.shape[0] or world_points.shape[0] != intrinsics.shape[0]:
        raise AssertionError('world_points, img_points and intrinsics must have the same batch size.')
    if world_points.shape[1] < 6:
        raise AssertionError(f'At least 6 points are required to use this function. Got {world_points.shape[1]} points.')
    (B, N) = world_points.shape[:2]
    (world_points_norm, world_transform_norm) = _mean_isotropic_scale_normalize(world_points)
    s = _torch_linalg_svdvals(world_points_norm)
    if torch.any(s[:, -1] < svd_eps):
        raise AssertionError(f'The last singular value of one/more of the elements of the batch is smaller than {svd_eps}. This function cannot be used if all world_points (of any element of the batch) lie on a line or if all world_points (of any element of the batch) lie on a plane.')
    intrinsics_inv = torch.inverse(intrinsics)
    world_points_norm_h = convert_points_to_homogeneous(world_points_norm)
    img_points_inv = transform_points(intrinsics_inv, img_points)
    (img_points_norm, img_transform_norm) = _mean_isotropic_scale_normalize(img_points_inv)
    inv_img_transform_norm = torch.inverse(img_transform_norm)
    system = torch.zeros((B, 2 * N, 12), dtype=world_points.dtype, device=world_points.device)
    system[:, 0::2, 0:4] = world_points_norm_h
    system[:, 1::2, 4:8] = world_points_norm_h
    system[:, 0::2, 8:12] = world_points_norm_h * -1 * img_points_norm[..., 0:1]
    system[:, 1::2, 8:12] = world_points_norm_h * -1 * img_points_norm[..., 1:2]
    (_, _, v) = torch.svd(system)
    solution = v[..., -1]
    solution = solution.reshape(B, 3, 4)
    solution_4x4 = eye_like(4, solution)
    solution_4x4[:, :3, :] = solution
    intermediate = torch.bmm(solution_4x4, world_transform_norm)
    solution = torch.bmm(inv_img_transform_norm, intermediate[:, :3, :])
    det = torch.det(solution[:, :3, :3])
    ones = torch.ones_like(det)
    sign_fix = torch.where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    temp = solution * mul_factor
    (ortho, right) = linalg_qr(temp[:, :3, :3])
    mask = eye_like(3, ortho)
    col_sign_fix = torch.sign(mask * right)
    rot_mat = torch.bmm(ortho, col_sign_fix)
    pred_world_to_cam = torch.cat([rot_mat, temp[:, :3, 3:4]], dim=-1)
    return pred_world_to_cam
import numba
import numpy as np
from .structures.utils import limit_period, points_cam2img, rotation_3d_in_axis

def camera_to_lidar(points, r_rect, velo2cam):
    if False:
        for i in range(10):
            print('nop')
    'Convert points in camera coordinate to lidar coordinate.\n\n    Note:\n        This function is for KITTI only.\n\n    Args:\n        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.\n        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in\n            specific camera coordinate (e.g. CAM2) to CAM0.\n        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in\n            camera coordinate to lidar coordinate.\n\n    Returns:\n        np.ndarray, shape=[N, 3]: Points in lidar coordinate.\n    '
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]

def box_camera_to_lidar(data, r_rect, velo2cam):
    if False:
        i = 10
        return i + 15
    'Convert boxes in camera coordinate to lidar coordinate.\n\n    Note:\n        This function is for KITTI only.\n\n    Args:\n        data (np.ndarray, shape=[N, 7]): Boxes in camera coordinate.\n        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in\n            specific camera coordinate (e.g. CAM2) to CAM0.\n        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in\n            camera coordinate to lidar coordinate.\n\n    Returns:\n        np.ndarray, shape=[N, 3]: Boxes in lidar coordinate.\n    '
    xyz = data[:, 0:3]
    (x_size, y_size, z_size) = (data[:, 3:4], data[:, 4:5], data[:, 5:6])
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    r_new = -r - np.pi / 2
    r_new = limit_period(r_new, period=np.pi * 2)
    return np.concatenate([xyz_lidar, x_size, z_size, y_size, r_new], axis=1)

def corners_nd(dims, origin=0.5):
    if False:
        for i in range(10):
            print('nop')
    'Generate relative box corners based on length per dim and origin point.\n\n    Args:\n        dims (np.ndarray, shape=[N, ndim]): Array of length per dim\n        origin (list or array or float, optional): origin point relate to\n            smallest point. Defaults to 0.5\n\n    Returns:\n        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.\n        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;\n            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1\n            where x0 < x1, y0 < y1, z0 < z1.\n    '
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    if False:
        while True:
            i = 10
    'Convert kitti locations, dimensions and angles to corners.\n    format: center(xy), dims(xy), angles(counterclockwise when positive)\n\n    Args:\n        centers (np.ndarray): Locations in kitti label file with shape (N, 2).\n        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).\n        angles (np.ndarray, optional): Rotation_y in kitti label file with\n            shape (N). Defaults to None.\n        origin (list or array or float, optional): origin point relate to\n            smallest point. Defaults to 0.5.\n\n    Returns:\n        np.ndarray: Corners with the shape of (N, 4, 2).\n    '
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

@numba.jit(nopython=True)
def depth_to_points(depth, trunc_pixel):
    if False:
        while True:
            i = 10
    'Convert depth map to points.\n\n    Args:\n        depth (np.array, shape=[H, W]): Depth map which\n            the row of [0~`trunc_pixel`] are truncated.\n        trunc_pixel (int): The number of truncated row.\n\n    Returns:\n        np.ndarray: Points in camera coordinates.\n    '
    num_pts = np.sum(depth[trunc_pixel:,] > 0.1)
    points = np.zeros((num_pts, 3), dtype=depth.dtype)
    x = np.array([0, 0, 1], dtype=depth.dtype)
    k = 0
    for i in range(trunc_pixel, depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i, j] > 0.1:
                x = np.array([j, i, 1], dtype=depth.dtype)
                points[k] = x * depth[i, j]
                k += 1
    return points

def depth_to_lidar_points(depth, trunc_pixel, P2, r_rect, velo2cam):
    if False:
        print('Hello World!')
    'Convert depth map to points in lidar coordinate.\n\n    Args:\n        depth (np.array, shape=[H, W]): Depth map which\n            the row of [0~`trunc_pixel`] are truncated.\n        trunc_pixel (int): The number of truncated row.\n        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.\n        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in\n            specific camera coordinate (e.g. CAM2) to CAM0.\n        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in\n            camera coordinate to lidar coordinate.\n\n    Returns:\n        np.ndarray: Points in lidar coordinates.\n    '
    pts = depth_to_points(depth, trunc_pixel)
    points_shape = list(pts.shape[0:-1])
    points = np.concatenate([pts, np.ones(points_shape + [1])], axis=-1)
    points = points @ np.linalg.inv(P2.T)
    lidar_points = camera_to_lidar(points, r_rect, velo2cam)
    return lidar_points

def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 1.0, 0.5), axis=1):
    if False:
        for i in range(10):
            print('nop')
    'Convert kitti locations, dimensions and angles to corners.\n\n    Args:\n        centers (np.ndarray): Locations in kitti label file with shape (N, 3).\n        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).\n        angles (np.ndarray, optional): Rotation_y in kitti label file with\n            shape (N). Defaults to None.\n        origin (list or array or float, optional): Origin point relate to\n            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)\n            in lidar. Defaults to (0.5, 1.0, 0.5).\n        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.\n            Defaults to 1.\n\n    Returns:\n        np.ndarray: Corners with the shape of (N, 8, 3).\n    '
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    if False:
        print('Hello World!')
    'Convert box2d to corner.\n\n    Args:\n        boxes (np.ndarray, shape=[N, 5]): Boxes2d with rotation.\n\n    Returns:\n        box_corners (np.ndarray, shape=[N, 4, 2]): Box corners.\n    '
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners

@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    if False:
        for i in range(10):
            print('nop')
    'Convert boxes_corner to aligned (min-max) boxes.\n\n    Args:\n        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.\n\n    Returns:\n        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.\n    '
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result

@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    if False:
        return 10
    'Convert 3d box corners from corner function above to surfaces that\n    normal vectors all direct to internal.\n\n    Args:\n        corners (np.ndarray): 3d box corners with the shape of (N, 8, 3).\n\n    Returns:\n        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).\n    '
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces

def rotation_points_single_angle(points, angle, axis=0):
    if False:
        return 10
    'Rotate points with a single angle.\n\n    Args:\n        points (np.ndarray, shape=[N, 3]]):\n        angle (np.ndarray, shape=[1]]):\n        axis (int, optional): Axis to rotate at. Defaults to 0.\n\n    Returns:\n        np.ndarray: Rotated points.\n    '
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]], dtype=points.dtype)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array([[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]], dtype=points.dtype)
    elif axis == 0:
        rot_mat_T = np.array([[1, 0, 0], [0, rot_cos, rot_sin], [0, -rot_sin, rot_cos]], dtype=points.dtype)
    else:
        raise ValueError('axis should in range')
    return (points @ rot_mat_T, rot_mat_T)

def box3d_to_bbox(box3d, P2):
    if False:
        i = 10
        return i + 15
    'Convert box3d in camera coordinates to bbox in image coordinates.\n\n    Args:\n        box3d (np.ndarray, shape=[N, 7]): Boxes in camera coordinate.\n        P2 (np.array, shape=[4, 4]): Intrinsics of Camera2.\n\n    Returns:\n        np.ndarray, shape=[N, 4]: Boxes 2d in image coordinates.\n    '
    box_corners = center_to_corner_box3d(box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1)
    box_corners_in_image = points_cam2img(box_corners, P2)
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox

def corner_to_surfaces_3d(corners):
    if False:
        i = 10
        return i + 15
    'convert 3d box corners from corner function above to surfaces that\n    normal vectors all direct to internal.\n\n    Args:\n        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).\n\n    Returns:\n        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).\n    '
    surfaces = np.array([[corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]], [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]], [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]], [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]], [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]], [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]]]).transpose([2, 0, 1, 3])
    return surfaces

def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    if False:
        print('Hello World!')
    'Check points in rotated bbox and return indices.\n\n    Note:\n        This function is for counterclockwise boxes.\n\n    Args:\n        points (np.ndarray, shape=[N, 3+dim]): Points to query.\n        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.\n        z_axis (int, optional): Indicate which axis is height.\n            Defaults to 2.\n        origin (tuple[int], optional): Indicate the position of\n            box center. Defaults to (0.5, 0.5, 0).\n\n    Returns:\n        np.ndarray, shape=[N, M]: Indices of points in each box.\n    '
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices

def minmax_to_corner_2d(minmax_box):
    if False:
        while True:
            i = 10
    'Convert minmax box to corners2d.\n\n    Args:\n        minmax_box (np.ndarray, shape=[N, dims]): minmax boxes.\n\n    Returns:\n        np.ndarray: 2d corners of boxes\n    '
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)

def create_anchors_3d_range(feature_size, anchor_range, sizes=((3.9, 1.6, 1.56),), rotations=(0, np.pi / 2), dtype=np.float32):
    if False:
        i = 10
        return i + 15
    'Create anchors 3d by range.\n\n    Args:\n        feature_size (list[float] | tuple[float]): Feature map size. It is\n            either a list of a tuple of [D, H, W](in order of z, y, and x).\n        anchor_range (torch.Tensor | list[float]): Range of anchors with\n            shape [6]. The order is consistent with that of anchors, i.e.,\n            (x_min, y_min, z_min, x_max, y_max, z_max).\n        sizes (list[list] | np.ndarray | torch.Tensor, optional):\n            Anchor size with shape [N, 3], in order of x, y, z.\n            Defaults to ((3.9, 1.6, 1.56), ).\n        rotations (list[float] | np.ndarray | torch.Tensor, optional):\n            Rotations of anchors in a single feature grid.\n            Defaults to (0, np.pi / 2).\n        dtype (type, optional): Data type. Defaults to np.float32.\n\n    Returns:\n        np.ndarray: Range based anchors with shape of\n            (*feature_size, num_sizes, num_rots, 7).\n    '
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])

def center_to_minmax_2d(centers, dims, origin=0.5):
    if False:
        i = 10
        return i + 15
    'Center to minmax.\n\n    Args:\n        centers (np.ndarray): Center points.\n        dims (np.ndarray): Dimensions.\n        origin (list or array or float, optional): Origin point relate\n            to smallest point. Defaults to 0.5.\n\n    Returns:\n        np.ndarray: Minmax points.\n    '
    if origin == 0.5:
        return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])

def rbbox2d_to_near_bbox(rbboxes):
    if False:
        while True:
            i = 10
    "convert rotated bbox to nearest 'standing' or 'lying' bbox.\n\n    Args:\n        rbboxes (np.ndarray): Rotated bboxes with shape of\n            (N, 5(x, y, xdim, ydim, rad)).\n\n    Returns:\n        np.ndarray: Bounding boxes with the shape of\n            (N, 4(xmin, ymin, xmax, ymax)).\n    "
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, mode='iou', eps=0.0):
    if False:
        i = 10
        return i + 15
    "Calculate box iou. Note that jit version runs ~10x faster than the\n    box_overlaps function in mmdet3d.core.evaluation.\n\n    Note:\n        This function is for counterclockwise boxes.\n\n    Args:\n        boxes (np.ndarray): Input bounding boxes with shape of (N, 4).\n        query_boxes (np.ndarray): Query boxes with shape of (K, 4).\n        mode (str, optional): IoU mode. Defaults to 'iou'.\n        eps (float, optional): Value added to denominator. Defaults to 0.\n\n    Returns:\n        np.ndarray: Overlap between boxes and query_boxes\n            with the shape of [N, K].\n    "
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (query_boxes[k, 3] - query_boxes[k, 1] + eps)
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + eps
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + eps
                if ih > 0:
                    if mode == 'iou':
                        ua = (boxes[n, 2] - boxes[n, 0] + eps) * (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih
                    else:
                        ua = (boxes[n, 2] - boxes[n, 0] + eps) * (boxes[n, 3] - boxes[n, 1] + eps)
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def projection_matrix_to_CRT_kitti(proj):
    if False:
        i = 10
        return i + 15
    'Split projection matrix of KITTI.\n\n    Note:\n        This function is for KITTI only.\n\n    P = C @ [R|T]\n    C is upper triangular matrix, so we need to inverse CR and use QR\n    stable for all kitti camera projection matrix.\n\n    Args:\n        proj (p.array, shape=[4, 4]): Intrinsics of camera.\n\n    Returns:\n        tuple[np.ndarray]: Splited matrix of C, R and T.\n    '
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    (Rinv, Cinv) = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return (C, R, T)

def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    if False:
        return 10
    'Remove points which are outside of image.\n\n    Note:\n        This function is for KITTI only.\n\n    Args:\n        points (np.ndarray, shape=[N, 3+dims]): Total points.\n        rect (np.ndarray, shape=[4, 4]): Matrix to project points in\n            specific camera coordinate (e.g. CAM2) to CAM0.\n        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in\n            camera coordinate to lidar coordinate.\n        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.\n        image_shape (list[int]): Shape of image.\n\n    Returns:\n        np.ndarray, shape=[N, 3+dims]: Filtered points.\n    '
    (C, R, T) = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points

def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    if False:
        print('Hello World!')
    'Get frustum corners in camera coordinates.\n\n    Args:\n        bbox_image (list[int]): box in image coordinates.\n        C (np.ndarray): Intrinsics.\n        near_clip (float, optional): Nearest distance of frustum.\n            Defaults to 0.001.\n        far_clip (float, optional): Farthest distance of frustum.\n            Defaults to 100.\n\n    Returns:\n        np.ndarray, shape=[8, 3]: coordinates of frustum corners.\n    '
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz

def surface_equ_3d(polygon_surfaces):
    if False:
        print('Hello World!')
    "\n\n    Args:\n        polygon_surfaces (np.ndarray): Polygon surfaces with shape of\n            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].\n            All surfaces' normal vector must direct to internal.\n            Max_num_points_of_surface must at least 3.\n\n    Returns:\n        tuple: normal vector and its direction.\n    "
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return (normal_vec, -d)

@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces):
    if False:
        print('Hello World!')
    "\n    Args:\n        points (np.ndarray): Input points with shape of (num_points, 3).\n        polygon_surfaces (np.ndarray): Polygon surfaces with shape of\n            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).\n            All surfaces' normal vector must direct to internal.\n            Max_num_points_of_surface must at least 3.\n        normal_vec (np.ndarray): Normal vector of polygon_surfaces.\n        d (int): Directions of normal vector.\n        num_surfaces (np.ndarray): Number of surfaces a polygon contains\n            shape of (num_polygon).\n\n    Returns:\n        np.ndarray: Result matrix with the shape of [num_points, num_polygon].\n    "
    (max_num_surfaces, max_num_points_of_surface) = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] + points[i, 1] * normal_vec[j, k, 1] + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    if False:
        i = 10
        return i + 15
    "Check points is in 3d convex polygons.\n\n    Args:\n        points (np.ndarray): Input points with shape of (num_points, 3).\n        polygon_surfaces (np.ndarray): Polygon surfaces with shape of\n            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).\n            All surfaces' normal vector must direct to internal.\n            Max_num_points_of_surface must at least 3.\n        num_surfaces (np.ndarray, optional): Number of surfaces a polygon\n            contains shape of (num_polygon). Defaults to None.\n\n    Returns:\n        np.ndarray: Result matrix with the shape of [num_points, num_polygon].\n    "
    (max_num_surfaces, max_num_points_of_surface) = polygon_surfaces.shape[1:3]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    (normal_vec, d) = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)

@numba.njit
def points_in_convex_polygon_jit(points, polygon, clockwise=False):
    if False:
        i = 10
        return i + 15
    'Check points is in 2d convex polygons. True when point in polygon.\n\n    Args:\n        points (np.ndarray): Input points with the shape of [num_points, 2].\n        polygon (np.ndarray): Input polygon with the shape of\n            [num_polygon, num_points_of_polygon, 2].\n        clockwise (bool, optional): Indicate polygon is clockwise. Defaults\n            to True.\n\n    Returns:\n        np.ndarray: Result matrix with the shape of [num_points, num_polygon].\n    '
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - polygon[:, np.array([num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1))), :]
    else:
        vec1 = polygon[:, np.array([num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1))), :] - polygon
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                vec = vec1[j, k]
                cross = vec[1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec[0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret

def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    if False:
        for i in range(10):
            print('nop')
    'Convert kitti center boxes to corners.\n\n        7 -------- 4\n       /|         /|\n      6 -------- 5 .\n      | |        | |\n      . 3 -------- 0\n      |/         |/\n      2 -------- 1\n\n    Note:\n        This function is for LiDAR boxes only.\n\n    Args:\n        boxes3d (np.ndarray): Boxes with shape of (N, 7)\n            [x, y, z, x_size, y_size, z_size, ry] in LiDAR coords,\n            see the definition of ry in KITTI dataset.\n        bottom_center (bool, optional): Whether z is on the bottom center\n            of object. Defaults to True.\n\n    Returns:\n        np.ndarray: Box corners with the shape of [N, 8, 3].\n    '
    boxes_num = boxes3d.shape[0]
    (x_size, y_size, z_size) = (boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5])
    x_corners = np.array([x_size / 2.0, -x_size / 2.0, -x_size / 2.0, x_size / 2.0, x_size / 2.0, -x_size / 2.0, -x_size / 2.0, x_size / 2.0], dtype=np.float32).T
    y_corners = np.array([-y_size / 2.0, -y_size / 2.0, y_size / 2.0, y_size / 2.0, -y_size / 2.0, -y_size / 2.0, y_size / 2.0, y_size / 2.0], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = z_size.reshape(boxes_num, 1).repeat(4, axis=1)
    else:
        z_corners = np.array([-z_size / 2.0, -z_size / 2.0, -z_size / 2.0, -z_size / 2.0, z_size / 2.0, z_size / 2.0, z_size / 2.0, z_size / 2.0], dtype=np.float32).T
    ry = boxes3d[:, 6]
    (zeros, ones) = (np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32))
    rot_list = np.array([[np.cos(ry), np.sin(ry), zeros], [-np.sin(ry), np.cos(ry), zeros], [zeros, zeros, ones]])
    R_list = np.transpose(rot_list, (2, 0, 1))
    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)), axis=2)
    rotated_corners = np.matmul(temp_corners, R_list)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]
    (x_loc, y_loc, z_loc) = (boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2])
    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)
    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)
    return corners.astype(np.float32)
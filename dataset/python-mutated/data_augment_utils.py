import warnings
import numba
import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from mmdet3d.core.bbox import box_np_ops
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    if False:
        while True:
            i = 10
    'Rotate 2D boxes.\n\n    Args:\n        corners (np.ndarray): Corners of boxes.\n        angle (float): Rotation angle.\n        rot_mat_T (np.ndarray): Transposed rotation matrix.\n    '
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = rot_sin
    rot_mat_T[1, 0] = -rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    if False:
        while True:
            i = 10
    'Box collision test.\n\n    Args:\n        boxes (np.ndarray): Corners of current boxes.\n        qboxes (np.ndarray): Boxes to be avoid colliding.\n        clockwise (bool, optional): Whether the corners are in\n            clockwise order. Default: True.\n    '
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(boxes_standup[i, 0], qboxes_standup[j, 0])
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(boxes_standup[i, 1], qboxes_standup[j, 1])
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        box_overlap_qbox = True
                        for box_l in range(4):
                            for k in range(4):
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break
                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):
                                for k in range(4):
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True
                        else:
                            ret[i, j] = True
    return ret

@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    if False:
        i = 10
        return i + 15
    'Add noise to every box (only on the horizontal plane).\n\n    Args:\n        boxes (np.ndarray): Input boxes with shape (N, 5).\n        valid_mask (np.ndarray): Mask to indicate which boxes are valid\n            with shape (N).\n        loc_noises (np.ndarray): Location noises with shape (N, M, 3).\n        rot_noises (np.ndarray): Rotation noises with shape (N, M).\n\n    Returns:\n        np.ndarray: Mask to indicate whether the noise is\n            added successfully (pass the collision test).\n    '
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask

@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    if False:
        i = 10
        return i + 15
    'Add noise to every box (only on the horizontal plane). Version 2 used\n    when enable global rotations.\n\n    Args:\n        boxes (np.ndarray): Input boxes with shape (N, 5).\n        valid_mask (np.ndarray): Mask to indicate which boxes are valid\n            with shape (N).\n        loc_noises (np.ndarray): Location noises with shape (N, M, 3).\n        rot_noises (np.ndarray): Rotation noises with shape (N, M).\n\n    Returns:\n        np.ndarray: Mask to indicate whether the noise is\n            added successfully (pass the collision test).\n    '
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot
                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = rot_sin
                rot_mat_T[1, 0] = -rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask

def _select_transform(transform, indices):
    if False:
        print('Hello World!')
    'Select transform.\n\n    Args:\n        transform (np.ndarray): Transforms to select from.\n        indices (np.ndarray): Mask to indicate which transform to select.\n\n    Returns:\n        np.ndarray: Selected transforms.\n    '
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result

@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    if False:
        while True:
            i = 10
    'Get the 3D rotation matrix.\n\n    Args:\n        rot_mat_T (np.ndarray): Transposed rotation matrix.\n        angle (float): Rotation angle.\n        axis (int): Rotation axis.\n    '
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = rot_sin
        rot_mat_T[2, 0] = -rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = rot_sin
        rot_mat_T[2, 1] = -rot_sin
        rot_mat_T[2, 2] = rot_cos

@numba.njit
def points_transform_(points, centers, point_masks, loc_transform, rot_transform, valid_mask):
    if False:
        return 10
    'Apply transforms to points and box centers.\n\n    Args:\n        points (np.ndarray): Input points.\n        centers (np.ndarray): Input box centers.\n        point_masks (np.ndarray): Mask to indicate which points need\n            to be transformed.\n        loc_transform (np.ndarray): Location transform to be applied.\n        rot_transform (np.ndarray): Rotation transform to be applied.\n        valid_mask (np.ndarray): Mask to indicate which boxes are valid.\n    '
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break

@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    if False:
        return 10
    'Transform 3D boxes.\n\n    Args:\n        boxes (np.ndarray): 3D boxes to be transformed.\n        loc_transform (np.ndarray): Location transform to be applied.\n        rot_transform (np.ndarray): Rotation transform to be applied.\n        valid_mask (np.ndarray): Mask to indicate which boxes are valid.\n    '
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]

def noise_per_object_v3_(gt_boxes, points=None, valid_mask=None, rotation_perturb=np.pi / 4, center_noise_std=1.0, global_random_rot_range=np.pi / 4, num_try=100):
    if False:
        print('Hello World!')
    'Random rotate or remove each groundtruth independently. use kitti viewer\n    to test this function points_transform_\n\n    Args:\n        gt_boxes (np.ndarray): Ground truth boxes with shape (N, 7).\n        points (np.ndarray, optional): Input point cloud with\n            shape (M, 4). Default: None.\n        valid_mask (np.ndarray, optional): Mask to indicate which\n            boxes are valid. Default: None.\n        rotation_perturb (float, optional): Rotation perturbation.\n            Default: pi / 4.\n        center_noise_std (float, optional): Center noise standard deviation.\n            Default: 1.0.\n        global_random_rot_range (float, optional): Global random rotation\n            range. Default: pi/4.\n        num_try (int, optional): Number of try. Default: 100.\n    '
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 0.001
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(grot_lowers[..., np.newaxis], grot_uppers[..., np.newaxis], size=[num_boxes, num_try])
    origin = (0.5, 0.5, 0)
    gt_box_corners = box_np_ops.center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2)
    if not enable_grot:
        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms, rot_transforms, valid_mask)
    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)
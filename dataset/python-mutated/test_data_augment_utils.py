import mmcv
import numpy as np
from mmdet3d.datasets.pipelines.data_augment_utils import noise_per_object_v3_, points_transform_

def test_noise_per_object_v3_():
    if False:
        return 10
    np.random.seed(0)
    points = np.fromfile('./tests/data/kitti/training/velodyne_reduced/000000.bin', np.float32).reshape(-1, 4)
    annos = mmcv.load('./tests/data/kitti/kitti_infos_train.pkl')
    info = annos[0]
    annos = info['annos']
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    noise_per_object_v3_(gt_boxes=gt_bboxes_3d, points=points)
    expected_gt_bboxes_3d = np.array([[3.3430212, 2.1475432, 9.388738, 1.2, 1.89, 0.48, 0.05056486]])
    assert points.shape == (800, 4)
    assert np.allclose(gt_bboxes_3d, expected_gt_bboxes_3d)

def test_points_transform():
    if False:
        i = 10
        return i + 15
    points = np.array([[46.509, 6.114, -0.779, 0.0], [42.949, 6.405, -0.705, 0.0], [42.901, 6.536, -0.705, 0.0], [46.196, 6.096, -1.01, 0.0], [43.308, 6.268, -0.936, 0.0]])
    gt_boxes = np.array([[15.34, 8.4691, -1.6855, 1.64, 3.7, 1.49, 3.13], [17.999, 8.2386, -1.5802, 1.55, 4.02, 1.52, 3.13], [29.62, 8.2617, -1.6185, 1.78, 4.25, 1.9, -3.12], [48.218, 7.8035, -1.379, 1.64, 3.7, 1.52, -0.01], [33.079, -8.4817, -1.3092, 0.43, 1.7, 1.62, -1.57]])
    point_masks = np.array([[False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]])
    loc_transforms = np.array([[-1.8635, -0.2774, -0.1774], [-1.0297, -1.0302, -0.3062], [1.668, 0.2597, 0.0551], [0.223, 0.7257, -0.0097], [-0.1403, 0.83, 0.3431]])
    rot_transforms = np.array([0.6888, -0.3858, 0.191, -0.0044, -0.0036])
    valid_mask = np.array([True, True, True, True, True])
    points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms, rot_transforms, valid_mask)
    assert points.shape == (5, 4)
    assert gt_boxes.shape == (5, 7)
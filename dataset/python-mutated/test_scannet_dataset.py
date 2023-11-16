import copy
import numpy as np
import pytest
import torch
from mmdet3d.datasets import ScanNetDataset, ScanNetInstanceSegDataset, ScanNetSegDataset

def test_getitem():
    if False:
        while True:
            i = 10
    np.random.seed(0)
    root_path = './tests/data/scannet/'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    pipelines = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=True, load_dim=6, use_dim=[0, 1, 2]), dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_mask_3d=True, with_seg_3d=True), dict(type='GlobalAlignment', rotation_axis=2), dict(type='PointSegClassMapping', valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)), dict(type='PointSample', num_points=5), dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=1.0), dict(type='GlobalRotScaleTrans', rot_range=[-0.087266, 0.087266], scale_ratio_range=[1.0, 1.0], shift_height=True), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'], meta_keys=['file_name', 'sample_idx', 'pcd_rotation'])]
    scannet_dataset = ScanNetDataset(root_path, ann_file, pipelines)
    data = scannet_dataset[0]
    points = data['points']._data
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    gt_labels = data['gt_labels_3d']._data
    pts_semantic_mask = data['pts_semantic_mask']._data
    pts_instance_mask = data['pts_instance_mask']._data
    file_name = data['img_metas']._data['file_name']
    pcd_rotation = data['img_metas']._data['pcd_rotation']
    sample_idx = data['img_metas']._data['sample_idx']
    expected_rotation = np.array([[0.99654, 0.08311407, 0.0], [-0.08311407, 0.99654, 0.0], [0.0, 0.0, 1.0]])
    assert file_name == './tests/data/scannet/points/scene0000_00.bin'
    assert np.allclose(pcd_rotation, expected_rotation, 0.001)
    assert sample_idx == 'scene0000_00'
    expected_points = torch.tensor([[1.8339, 2.1093, 2.29, 2.3895], [3.6079, 0.14592, 2.0687, 2.1682], [4.1886, 5.0614, -0.10841, -0.0088736], [6.879, 1.5086, -0.093154, 0.0063816], [4.8253, 0.26668, 1.4917, 1.5912]])
    expected_gt_bboxes_3d = torch.tensor([[-1.1835, -3.6317, 1.5704, 1.7577, 0.3761, 0.5724, 0.0], [-3.1832, 3.2269, 1.1911, 0.6727, 0.2251, 0.6715, 0.0], [-0.9598, -2.2864, 0.0093, 0.7506, 2.5709, 1.2145, 0.0], [-2.6988, -2.7354, 0.8288, 0.768, 1.8877, 0.287, 0.0], [3.2989, 0.2885, -0.009, 0.76, 3.8814, 2.1603, 0.0]])
    expected_gt_labels = np.array([6, 6, 4, 9, 11, 11, 10, 0, 15, 17, 17, 17, 3, 12, 4, 4, 14, 1, 0, 0, 0, 0, 0, 0, 5, 5, 5])
    expected_pts_semantic_mask = np.array([0, 18, 18, 18, 18])
    expected_pts_instance_mask = np.array([44, 22, 10, 10, 57])
    original_classes = scannet_dataset.CLASSES
    assert scannet_dataset.CLASSES == class_names
    assert torch.allclose(points, expected_points, 0.01)
    assert gt_bboxes_3d.tensor[:5].shape == (5, 7)
    assert torch.allclose(gt_bboxes_3d.tensor[:5], expected_gt_bboxes_3d, 0.01)
    assert np.all(gt_labels.numpy() == expected_gt_labels)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_pts_instance_mask)
    assert original_classes == class_names
    scannet_dataset = ScanNetDataset(root_path, ann_file, pipeline=None, classes=['cabinet', 'bed'])
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'bed']
    scannet_dataset = ScanNetDataset(root_path, ann_file, pipeline=None, classes=('cabinet', 'bed'))
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ('cabinet', 'bed')
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + 'classes.txt'
        with open(path, 'w') as f:
            f.write('cabinet\nbed\n')
    scannet_dataset = ScanNetDataset(root_path, ann_file, pipeline=None, classes=path)
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'bed']

def test_evaluate():
    if False:
        for i in range(10):
            print('nop')
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetDataset(root_path, ann_file)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = DepthInstance3DBoxes(torch.tensor([[1.4813, 3.5207, 1.5704, 1.7445, 0.23196, 0.57235, 0.0], [2.904, -3.4803, 1.1911, 0.66078, 0.17072, 0.67154, 0.0], [1.1466, 2.1987, 0.0092576, 0.54184, 2.5346, 1.2145, 0.0], [2.9168, 2.5016, 0.82875, 0.61697, 1.8428, 0.28697, 0.0], [-3.3114, -0.013351, -0.0089524, 0.44082, 3.8582, 2.1603, 0.0], [-2.0135, -3.4857, 0.93848, 1.9911, 0.21603, 1.2767, 0.0], [-2.1945, -3.1402, -0.038165, 1.4801, 0.68676, 1.0586, 0.0], [-2.7553, 2.4055, -0.029972, 1.4764, 1.4927, 2.338, 0.0]]))
    pred_boxes['labels_3d'] = torch.tensor([6, 6, 4, 9, 11, 11])
    pred_boxes['scores_3d'] = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 0.5])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ret_dict = scannet_dataset.evaluate(results, metric)
    assert abs(ret_dict['table_AP_0.25'] - 0.3333) < 0.01
    assert abs(ret_dict['window_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['counter_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['curtain_AP_0.25'] - 1.0) < 0.01
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    eval_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, load_dim=6, use_dim=[0, 1, 2]), dict(type='GlobalAlignment', rotation_axis=2), dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False), dict(type='Collect3D', keys=['points'])]
    ret_dict = scannet_dataset.evaluate(results, metric, pipeline=eval_pipeline)
    assert abs(ret_dict['table_AP_0.25'] - 0.3333) < 0.01
    assert abs(ret_dict['window_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['counter_AP_0.25'] - 1.0) < 0.01
    assert abs(ret_dict['curtain_AP_0.25'] - 1.0) < 0.01

def test_show():
    if False:
        return 10
    import tempfile
    from os import path as osp
    import mmcv
    from mmdet3d.core.bbox import DepthInstance3DBoxes
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetDataset(root_path, ann_file)
    boxes_3d = DepthInstance3DBoxes(torch.tensor([[-2.4053, 0.92295, 0.080661, 2.4054, 2.1468, 0.8599, 0.0], [-1.9341, -2.0741, 0.0030698, 0.32206, 0.25322, 0.35144, 0.0], [-3.6908, 0.0080684, 0.26201, 0.41515, 0.76489, 0.53585, 0.0], [2.6332, 0.85143, -0.0049964, 0.30367, 1.3448, 1.8329, 0.0], [0.020221, 2.6153, 0.015109, 0.73335, 1.0429, 1.0251, 0.0]]))
    scores_3d = torch.tensor([0.00012058, 0.0023012, 6.2324e-06, 6.6139e-06, 6.7965e-05])
    labels_3d = torch.tensor([0, 0, 0, 0, 0])
    result = dict(boxes_3d=boxes_3d, scores_3d=scores_3d, labels_3d=labels_3d)
    results = [result]
    scannet_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_points.obj')
    gt_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_gt.obj')
    pred_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    eval_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, load_dim=6, use_dim=[0, 1, 2]), dict(type='GlobalAlignment', rotation_axis=2), dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False), dict(type='Collect3D', keys=['points'])]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    scannet_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_points.obj')
    gt_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_gt.obj')
    pred_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

def test_seg_getitem():
    if False:
        print('Hello World!')
    np.random.seed(0)
    root_path = './tests/data/scannet/'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    palette = [[174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120], [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141], [255, 127, 14], [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163]]
    scene_idxs = [0 for _ in range(20)]
    pipelines = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=False, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39), max_cat_id=40), dict(type='IndoorPatchPointSample', num_points=5, block_size=1.5, ignore_index=len(class_names), use_normalized_coord=True, enlarge_size=0.2, min_unique_num=None), dict(type='NormalizePointsColor', color_mean=None), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'pts_semantic_mask'], meta_keys=['file_name', 'sample_idx'])]
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=pipelines, classes=None, palette=None, modality=None, test_mode=False, ignore_index=None, scene_idxs=scene_idxs)
    data = scannet_dataset[0]
    points = data['points']._data
    pts_semantic_mask = data['pts_semantic_mask']._data
    file_name = data['img_metas']._data['file_name']
    sample_idx = data['img_metas']._data['sample_idx']
    assert file_name == './tests/data/scannet/points/scene0000_00.bin'
    assert sample_idx == 'scene0000_00'
    expected_points = torch.tensor([[0.0, 0.0, 1.2427, 0.6118, 0.5529, 0.4471, -0.6462, -1.0046, 0.428], [0.1553, -0.0074, 1.6077, 0.5882, 0.6157, 0.5569, -0.6001, -1.0068, 0.5537], [0.1518, 0.6016, 0.6548, 0.149, 0.1059, 0.0431, -0.6012, -0.8309, 0.2255], [-0.7494, 0.1033, 0.6756, 0.5216, 0.4353, 0.3333, -0.8687, -0.9748, 0.2327], [-0.6836, -0.0203, 0.5884, 0.5765, 0.502, 0.451, -0.8491, -1.0105, 0.2027]])
    expected_pts_semantic_mask = np.array([13, 13, 12, 2, 0])
    original_classes = scannet_dataset.CLASSES
    original_palette = scannet_dataset.PALETTE
    assert scannet_dataset.CLASSES == class_names
    assert scannet_dataset.ignore_index == 20
    assert torch.allclose(points, expected_points, 0.01)
    assert np.all(pts_semantic_mask.numpy() == expected_pts_semantic_mask)
    assert original_classes == class_names
    assert original_palette == palette
    assert scannet_dataset.scene_idxs.dtype == np.int32
    assert np.all(scannet_dataset.scene_idxs == np.array(scene_idxs))
    np.random.seed(0)
    new_pipelines = copy.deepcopy(pipelines)
    new_pipelines[3] = dict(type='IndoorPatchPointSample', num_points=5, block_size=1.5, ignore_index=len(class_names), use_normalized_coord=False, enlarge_size=0.2, min_unique_num=None)
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=new_pipelines, scene_idxs=scene_idxs)
    data = scannet_dataset[0]
    points = data['points']._data
    assert torch.allclose(points, expected_points[:, :6], 0.01)
    np.random.seed(0)
    new_pipelines = copy.deepcopy(pipelines)
    new_pipelines[0] = dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=False, load_dim=6, use_dim=[0, 1, 2])
    new_pipelines.remove(new_pipelines[4])
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=new_pipelines, scene_idxs=scene_idxs)
    data = scannet_dataset[0]
    points = data['points']._data
    assert torch.allclose(points, expected_points[:, [0, 1, 2, 6, 7, 8]], 0.01)
    np.random.seed(0)
    new_pipelines = copy.deepcopy(pipelines)
    new_pipelines[0] = dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=False, load_dim=6, use_dim=[0, 1, 2])
    new_pipelines[3] = dict(type='IndoorPatchPointSample', num_points=5, block_size=1.5, ignore_index=len(class_names), use_normalized_coord=False, enlarge_size=0.2, min_unique_num=None)
    new_pipelines.remove(new_pipelines[4])
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=new_pipelines, scene_idxs=scene_idxs)
    data = scannet_dataset[0]
    points = data['points']._data
    assert torch.allclose(points, expected_points[:, :3], 0.01)
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=None, classes=['cabinet', 'chair'], scene_idxs=scene_idxs)
    label_map = {i: 20 for i in range(41)}
    label_map.update({3: 0, 5: 1})
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'chair']
    assert scannet_dataset.PALETTE == [palette[2], palette[4]]
    assert scannet_dataset.VALID_CLASS_IDS == [3, 5]
    assert scannet_dataset.label_map == label_map
    assert scannet_dataset.label2cat == {0: 'cabinet', 1: 'chair'}
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + 'classes.txt'
        with open(path, 'w') as f:
            f.write('cabinet\nchair\n')
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=None, classes=path, scene_idxs=scene_idxs)
    assert scannet_dataset.CLASSES != original_classes
    assert scannet_dataset.CLASSES == ['cabinet', 'chair']
    assert scannet_dataset.PALETTE == [palette[2], palette[4]]
    assert scannet_dataset.VALID_CLASS_IDS == [3, 5]
    assert scannet_dataset.label_map == label_map
    assert scannet_dataset.label2cat == {0: 'cabinet', 1: 'chair'}
    with pytest.raises(NotImplementedError):
        scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=None, scene_idxs=None)
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, pipeline=None, test_mode=True, scene_idxs=scene_idxs)
    assert np.all(scannet_dataset.scene_idxs == np.array([0]))

def test_seg_evaluate():
    if False:
        return 10
    if not torch.cuda.is_available():
        pytest.skip()
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, test_mode=True)
    results = []
    pred_sem_mask = dict(semantic_mask=torch.tensor([13, 5, 1, 2, 6, 2, 13, 1, 14, 2, 0, 0, 5, 5, 3, 0, 1, 14, 0, 0, 0, 18, 6, 15, 13, 0, 2, 4, 0, 3, 16, 6, 13, 5, 13, 0, 0, 0, 0, 1, 7, 3, 19, 12, 8, 0, 11, 0, 0, 1, 2, 13, 17, 1, 1, 1, 6, 2, 13, 19, 4, 17, 0, 14, 1, 7, 2, 1, 7, 2, 0, 5, 17, 5, 0, 0, 3, 6, 5, 11, 1, 13, 13, 2, 3, 1, 0, 13, 19, 1, 14, 5, 3, 1, 13, 1, 2, 3, 2, 1]).long())
    results.append(pred_sem_mask)
    class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    eval_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=False, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39), max_cat_id=40), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])]
    ret_dict = scannet_dataset.evaluate(results, pipeline=eval_pipeline)
    assert abs(ret_dict['miou'] - 0.5308) < 0.01
    assert abs(ret_dict['acc'] - 0.8219) < 0.01
    assert abs(ret_dict['acc_cls'] - 0.7649) < 0.01

def test_seg_show():
    if False:
        i = 10
        return i + 15
    import tempfile
    from os import path as osp
    import mmcv
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, scene_idxs=[0])
    result = dict(semantic_mask=torch.tensor([13, 5, 1, 2, 6, 2, 13, 1, 14, 2, 0, 0, 5, 5, 3, 0, 1, 14, 0, 0, 0, 18, 6, 15, 13, 0, 2, 4, 0, 3, 16, 6, 13, 5, 13, 0, 0, 0, 0, 1, 7, 3, 19, 12, 8, 0, 11, 0, 0, 1, 2, 13, 17, 1, 1, 1, 6, 2, 13, 19, 4, 17, 0, 14, 1, 7, 2, 1, 7, 2, 0, 5, 17, 5, 0, 0, 3, 6, 5, 11, 1, 13, 13, 2, 3, 1, 0, 13, 19, 1, 14, 5, 3, 1, 13, 1, 2, 3, 2, 1]).long())
    results = [result]
    scannet_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_points.obj')
    gt_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_gt.obj')
    pred_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    eval_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=False, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39), max_cat_id=40), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])]
    scannet_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_points.obj')
    gt_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_gt.obj')
    pred_file_path = osp.join(temp_dir, 'scene0000_00', 'scene0000_00_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

def test_seg_format_results():
    if False:
        print('Hello World!')
    from os import path as osp
    import mmcv
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    scannet_dataset = ScanNetSegDataset(data_root=root_path, ann_file=ann_file, test_mode=True)
    results = []
    pred_sem_mask = dict(semantic_mask=torch.tensor([13, 5, 1, 2, 6, 2, 13, 1, 14, 2, 0, 0, 5, 5, 3, 0, 1, 14, 0, 0, 0, 18, 6, 15, 13, 0, 2, 4, 0, 3, 16, 6, 13, 5, 13, 0, 0, 0, 0, 1, 7, 3, 19, 12, 8, 0, 11, 0, 0, 1, 2, 13, 17, 1, 1, 1, 6, 2, 13, 19, 4, 17, 0, 14, 1, 7, 2, 1, 7, 2, 0, 5, 17, 5, 0, 0, 3, 6, 5, 11, 1, 13, 13, 2, 3, 1, 0, 13, 19, 1, 14, 5, 3, 1, 13, 1, 2, 3, 2, 1]).long())
    results.append(pred_sem_mask)
    (result_files, tmp_dir) = scannet_dataset.format_results(results)
    expected_label = np.array([16, 6, 2, 3, 7, 3, 16, 2, 24, 3, 1, 1, 6, 6, 4, 1, 2, 24, 1, 1, 1, 36, 7, 28, 16, 1, 3, 5, 1, 4, 33, 7, 16, 6, 16, 1, 1, 1, 1, 2, 8, 4, 39, 14, 9, 1, 12, 1, 1, 2, 3, 16, 34, 2, 2, 2, 7, 3, 16, 39, 5, 34, 1, 24, 2, 8, 3, 2, 8, 3, 1, 6, 34, 6, 1, 1, 4, 7, 6, 12, 2, 16, 16, 3, 4, 2, 1, 16, 39, 2, 24, 6, 4, 2, 16, 2, 3, 4, 3, 2])
    expected_txt_path = osp.join(tmp_dir.name, 'results', 'scene0000_00.txt')
    assert np.all(result_files[0]['seg_mask'] == expected_label)
    mmcv.check_file_exist(expected_txt_path)

def test_instance_seg_getitem():
    if False:
        while True:
            i = 10
    np.random.seed(0)
    root_path = './tests/data/scannet/'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    train_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=True, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39), max_cat_id=40), dict(type='NormalizePointsColor', color_mean=None), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])]
    scannet_dataset = ScanNetInstanceSegDataset(data_root=root_path, ann_file=ann_file, pipeline=train_pipeline, classes=class_names, test_mode=False)
    expected_points = torch.tensor([[-3.4742, 0.78792, 1.7397, 0.33725, 0.35294, 0.30588], [2.7216, 3.4164, 2.4572, 0.66275, 0.62745, 0.51373], [1.3404, -1.4675, -0.044059, 0.38431, 0.36078, 0.35686], [-3.0335, 2.7273, 1.5181, 0.23137, 0.16078, 0.082353], [-0.43207, 1.8154, 0.17455, 0.40392, 0.38039, 0.41961]])
    data = scannet_dataset[0]
    points = data['points']._data[:5]
    pts_semantic_mask = data['pts_semantic_mask']._data[:5]
    pts_instance_mask = data['pts_instance_mask']._data[:5]
    expected_semantic_mask = np.array([11, 18, 18, 0, 4])
    expected_instance_mask = np.array([6, 56, 10, 9, 35])
    assert torch.allclose(points, expected_points, 0.01)
    assert np.all(pts_semantic_mask.numpy() == expected_semantic_mask)
    assert np.all(pts_instance_mask.numpy() == expected_instance_mask)

def test_instance_seg_evaluate():
    if False:
        i = 10
        return i + 15
    root_path = './tests/data/scannet'
    ann_file = './tests/data/scannet/scannet_infos.pkl'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    test_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='NormalizePointsColor', color_mean=None), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points'])]
    scannet_dataset = ScanNetInstanceSegDataset(data_root=root_path, ann_file=ann_file, pipeline=test_pipeline, test_mode=True)
    pred_mask = torch.tensor([1, -1, -1, -1, 7, 11, 2, -1, 1, 10, -1, -1, 5, -1, -1, -1, -1, 1, -1, -1, -1, -1, 0, -1, 1, -1, 12, -1, -1, -1, 8, 5, 1, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 8, -1, -1, -1, 0, 4, 3, -1, 9, -1, -1, 6, -1, -1, -1, -1, 13, -1, -1, 5, -1, 5, -1, -1, 9, 0, 5, -1, -1, 2, 3, 4, -1, -1, -1, 2, -1, -1, -1, 5, 9, -1, 1, -1, 4, 10, 4, -1]).long()
    pred_labels = torch.tensor([4, 11, 11, 10, 0, 3, 12, 4, 14, 1, 0, 0, 0, 5, 5]).long()
    pred_scores = torch.tensor([0.99 for _ in range(len(pred_labels))])
    results = [dict(instance_mask=pred_mask, instance_label=pred_labels, instance_score=torch.tensor(pred_scores))]
    eval_pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=True, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39), max_cat_id=40), dict(type='NormalizePointsColor', color_mean=None), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])]
    ret_dict = scannet_dataset.evaluate(results, pipeline=eval_pipeline, options=dict(min_region_sizes=np.array([1])))
    assert abs(ret_dict['all_ap'] - 0.90625) < 0.001
    assert abs(ret_dict['all_ap_50%'] - 0.90625) < 0.001
    assert abs(ret_dict['all_ap_25%'] - 0.94444) < 0.001
    assert abs(ret_dict['classes']['cabinet']['ap25%'] - 1.0) < 0.001
    assert abs(ret_dict['classes']['cabinet']['ap50%'] - 0.65625) < 0.001
    assert abs(ret_dict['classes']['door']['ap25%'] - 0.5) < 0.001
    assert abs(ret_dict['classes']['door']['ap50%'] - 0.5) < 0.001
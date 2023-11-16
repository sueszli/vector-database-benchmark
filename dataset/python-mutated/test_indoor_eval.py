import numpy as np
import pytest
import torch
from mmdet3d.core.evaluation.indoor_eval import average_precision, indoor_eval

def test_indoor_eval():
    if False:
        i = 10
        return i + 15
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import Box3DMode, DepthInstance3DBoxes
    det_infos = [{'labels_3d': torch.tensor([0, 1, 2, 2, 0, 3, 1, 2, 3, 2]), 'boxes_3d': DepthInstance3DBoxes(torch.tensor([[-0.0024089, -3.3174, 0.49438, 2.1668, 0.28431, 1.6506, 0.0], [-0.34269, -2.7565, 0.028144, 0.68554, 0.96854, 0.61755, 0.0], [-3.832, -1.0646, 0.17074, 0.24981, 0.44708, 0.62538, 0.0], [0.41073, 3.3757, 0.34311, 0.80617, 0.28679, 1.606, 0.0], [0.61199, -3.1041, 0.41873, 1.231, 0.40162, 1.7303, 0.0], [-0.59877, -2.6011, 1.1148, 0.15704, 0.75957, 0.9693, 0.0], [0.27462, -3.0088, 0.065231, 0.81208, 0.41861, 0.37339, 0.0], [-1.4704, -2.0024, 0.27479, 1.7888, 1.0566, 1.3704, 0.0], [0.082727, -3.116, 0.2569, 1.4054, 0.20772, 0.96792, 0.0], [2.6896, 1.9881, 1.1566, 0.099885, 0.35713, 0.45638, 0.0]]), origin=(0.5, 0.5, 0)), 'scores_3d': torch.tensor([1.7516e-05, 1.0167e-06, 8.4486e-07, 0.071048, 6.4274e-05, 1.5003e-07, 5.8102e-06, 1.9399e-08, 5.3126e-07, 1.863e-09])}]
    label2cat = {0: 'cabinet', 1: 'bed', 2: 'chair', 3: 'sofa'}
    gt_annos = [{'gt_num': 10, 'gt_boxes_upright_depth': np.array([[-0.0024089, -3.3174, 0.49438, 2.1668, 0.28431, 1.6506, 0.0], [-0.34269, -2.7565, 0.028144, 0.68554, 0.96854, 0.61755, 0.0], [-3.832, -1.0646, 0.17074, 0.24981, 0.44708, 0.62538, 0.0], [0.41073, 3.3757, 0.34311, 0.80617, 0.28679, 1.606, 0.0], [0.61199, -3.1041, 0.41873, 1.231, 0.40162, 1.7303, 0.0], [-0.59877, -2.6011, 1.1148, 0.15704, 0.75957, 0.9693, 0.0], [0.27462, -3.0088, 0.065231, 0.81208, 0.41861, 0.37339, 0.0], [-1.4704, -2.0024, 0.27479, 1.7888, 1.0566, 1.3704, 0.0], [0.082727, -3.116, 0.2569, 1.4054, 0.20772, 0.96792, 0.0], [2.6896, 1.9881, 1.1566, 0.099885, 0.35713, 0.45638, 0.0]]), 'class': np.array([0, 1, 2, 0, 0, 3, 1, 3, 3, 2])}]
    ret_value = indoor_eval(gt_annos, det_infos, [0.25, 0.5], label2cat, box_type_3d=DepthInstance3DBoxes, box_mode_3d=Box3DMode.DEPTH)
    assert np.isclose(ret_value['cabinet_AP_0.25'], 0.666667)
    assert np.isclose(ret_value['bed_AP_0.25'], 1.0)
    assert np.isclose(ret_value['chair_AP_0.25'], 0.5)
    assert np.isclose(ret_value['mAP_0.25'], 0.708333)
    assert np.isclose(ret_value['mAR_0.25'], 0.833333)

def test_indoor_eval_less_classes():
    if False:
        print('Hello World!')
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import Box3DMode, DepthInstance3DBoxes
    det_infos = [{'labels_3d': torch.tensor([0]), 'boxes_3d': DepthInstance3DBoxes(torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])), 'scores_3d': torch.tensor([0.5])}, {'labels_3d': torch.tensor([1]), 'boxes_3d': DepthInstance3DBoxes(torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])), 'scores_3d': torch.tensor([0.5])}]
    label2cat = {0: 'cabinet', 1: 'bed', 2: 'chair'}
    gt_annos = [{'gt_num': 2, 'gt_boxes_upright_depth': np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]), 'class': np.array([2, 0])}, {'gt_num': 1, 'gt_boxes_upright_depth': np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]), 'class': np.array([1])}]
    ret_value = indoor_eval(gt_annos, det_infos, [0.25, 0.5], label2cat, box_type_3d=DepthInstance3DBoxes, box_mode_3d=Box3DMode.DEPTH)
    assert np.isclose(ret_value['mAP_0.25'], 0.666667)
    assert np.isclose(ret_value['mAR_0.25'], 0.666667)

def test_average_precision():
    if False:
        while True:
            i = 10
    ap = average_precision(np.array([[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]), np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), '11points')
    assert abs(ap[0] - 0.06611571) < 0.001
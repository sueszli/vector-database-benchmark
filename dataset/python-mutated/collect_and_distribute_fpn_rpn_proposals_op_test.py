import numpy as np
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st
from caffe2.python import core, utils
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

def boxes_area(boxes):
    if False:
        i = 10
        return i + 15
    'Compute the area of an array of boxes.'
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    areas = w * h
    assert np.all(areas >= 0), 'Negative areas founds'
    return areas

def map_rois_to_fpn_levels(rois, k_min, k_max, roi_canonical_scale, roi_canonical_level):
    if False:
        return 10
    'Determine which FPN level each RoI in a set of RoIs should map to based\n    on the heuristic in the FPN paper.\n    '
    s = np.sqrt(boxes_area(rois))
    target_lvls = np.floor(roi_canonical_level + np.log2(s / roi_canonical_scale + 1e-06))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls

def collect(inputs, **args):
    if False:
        return 10
    post_nms_topN = args['rpn_post_nms_topN']
    num_lvls = args['rpn_num_levels']
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]
    rois = np.concatenate(roi_inputs)
    scores = np.concatenate(score_inputs).squeeze()
    assert rois.shape[0] == scores.shape[0]
    inds = np.argsort(-scores, kind='mergesort')[:post_nms_topN]
    rois = rois[inds, :]
    return rois

def distribute(rois, _, outputs, **args):
    if False:
        print('Hello World!')
    'To understand the output blob order see return value of\n    roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)\n    '
    lvl_min = args['roi_min_level']
    lvl_max = lvl_min + args['roi_num_levels'] - 1
    lvls = map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max, args['roi_canonical_scale'], args['roi_canonical_level'])
    outputs[0] = rois
    rois_idx_order = np.empty((0,))
    for (output_idx, lvl) in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order, kind='mergesort')
    outputs[-1] = rois_idx_restore.astype(np.int32)

def collect_and_distribute_fpn_rpn_ref(*inputs):
    if False:
        while True:
            i = 10
    assert inputs
    args = inputs[-1]
    inputs = inputs[:-1]
    num_rpn_lvls = args['rpn_num_levels']
    assert len(inputs) == 2 * num_rpn_lvls
    N = inputs[0].shape[0]
    for i in range(num_rpn_lvls):
        assert len(inputs[i].shape) == 2
        assert inputs[i].shape[0] == N
        assert inputs[i].shape[1] == 5
    for i in range(num_rpn_lvls, 2 * num_rpn_lvls):
        assert len(inputs[i].shape) == 1
        assert inputs[i].shape[0] == N
    num_roi_lvls = args['roi_num_levels']
    outputs = (num_roi_lvls + 2) * [None]
    rois = collect(inputs, **args)
    distribute(rois, None, outputs, **args)
    return outputs

def collect_rpn_ref(*inputs):
    if False:
        print('Hello World!')
    args = inputs[-1]
    inputs = inputs[:-1]
    rois = collect(inputs, **args)
    return [rois]

def distribute_fpn_ref(*inputs):
    if False:
        return 10
    args = inputs[-1]
    inputs = inputs[:-1]
    rois = inputs[0]
    num_roi_lvls = args['roi_num_levels']
    outputs = (num_roi_lvls + 2) * [None]
    distribute(rois, None, outputs, **args)
    outputs.pop(0)
    return outputs

class TestCollectAndDistributeFpnRpnProposals(serial.SerializedTestCase):

    @staticmethod
    def _create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale):
        if False:
            print('Hello World!')
        np.random.seed(0)
        input_names = []
        inputs = []
        for lvl in range(rpn_num_levels):
            rpn_roi = roi_canonical_scale * np.random.rand(proposal_count, 5).astype(np.float32)
            for i in range(proposal_count):
                rpn_roi[i][3] += rpn_roi[i][1]
                rpn_roi[i][4] += rpn_roi[i][2]
            input_names.append('rpn_rois_fpn{}'.format(lvl + rpn_min_level))
            inputs.append(rpn_roi)
        for lvl in range(rpn_num_levels):
            rpn_roi_score = np.random.rand(proposal_count).astype(np.float32)
            input_names.append('rpn_roi_probs_fpn{}'.format(lvl + rpn_min_level))
            inputs.append(rpn_roi_score)
        return (input_names, inputs)

    @given(proposal_count=st.integers(min_value=1000, max_value=8000), rpn_min_level=st.integers(min_value=1, max_value=4), rpn_num_levels=st.integers(min_value=1, max_value=6), roi_min_level=st.integers(min_value=1, max_value=4), roi_num_levels=st.integers(min_value=1, max_value=6), rpn_post_nms_topN=st.integers(min_value=1000, max_value=4000), roi_canonical_scale=st.integers(min_value=100, max_value=300), roi_canonical_level=st.integers(min_value=1, max_value=8), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_collect_and_dist(self, proposal_count, rpn_min_level, rpn_num_levels, roi_min_level, roi_num_levels, rpn_post_nms_topN, roi_canonical_scale, roi_canonical_level, gc, dc):
        if False:
            while True:
                i = 10
        (input_names, inputs) = self._create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale)
        output_names = ['rois']
        for lvl in range(roi_num_levels):
            output_names.append('rois_fpn{}'.format(lvl + roi_min_level))
        output_names.append('rois_idx_restore')
        op = core.CreateOperator('CollectAndDistributeFpnRpnProposals', input_names, output_names, arg=[utils.MakeArgument('roi_canonical_scale', roi_canonical_scale), utils.MakeArgument('roi_canonical_level', roi_canonical_level), utils.MakeArgument('roi_max_level', roi_min_level + roi_num_levels - 1), utils.MakeArgument('roi_min_level', roi_min_level), utils.MakeArgument('rpn_max_level', rpn_min_level + rpn_num_levels - 1), utils.MakeArgument('rpn_min_level', rpn_min_level), utils.MakeArgument('rpn_post_nms_topN', rpn_post_nms_topN)], device_option=gc)
        args = {'rpn_min_level': rpn_min_level, 'rpn_num_levels': rpn_num_levels, 'roi_min_level': roi_min_level, 'roi_num_levels': roi_num_levels, 'rpn_post_nms_topN': rpn_post_nms_topN, 'roi_canonical_scale': roi_canonical_scale, 'roi_canonical_level': roi_canonical_level}
        self.assertReferenceChecks(device_option=gc, op=op, inputs=inputs + [args], reference=collect_and_distribute_fpn_rpn_ref)

    @given(proposal_count=st.integers(min_value=1000, max_value=8000), rpn_min_level=st.integers(min_value=1, max_value=4), rpn_num_levels=st.integers(min_value=1, max_value=6), roi_min_level=st.integers(min_value=1, max_value=4), roi_num_levels=st.integers(min_value=1, max_value=6), rpn_post_nms_topN=st.integers(min_value=1000, max_value=4000), roi_canonical_scale=st.integers(min_value=100, max_value=300), roi_canonical_level=st.integers(min_value=1, max_value=8), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_collect_and_dist_separately(self, proposal_count, rpn_min_level, rpn_num_levels, roi_min_level, roi_num_levels, rpn_post_nms_topN, roi_canonical_scale, roi_canonical_level, gc, dc):
        if False:
            while True:
                i = 10
        (input_names, inputs) = self._create_input(proposal_count, rpn_min_level, rpn_num_levels, roi_canonical_scale)
        collect_op = core.CreateOperator('CollectRpnProposals', input_names, ['rois'], arg=[utils.MakeArgument('rpn_max_level', rpn_min_level + rpn_num_levels - 1), utils.MakeArgument('rpn_min_level', rpn_min_level), utils.MakeArgument('rpn_post_nms_topN', rpn_post_nms_topN)], device_option=gc)
        collect_args = {'rpn_min_level': rpn_min_level, 'rpn_num_levels': rpn_num_levels, 'rpn_post_nms_topN': rpn_post_nms_topN}
        self.assertReferenceChecks(device_option=gc, op=collect_op, inputs=inputs + [collect_args], reference=collect_rpn_ref)
        rois = collect(inputs, **collect_args)
        output_names = []
        for lvl in range(roi_num_levels):
            output_names.append('rois_fpn{}'.format(lvl + roi_min_level))
        output_names.append('rois_idx_restore')
        distribute_op = core.CreateOperator('DistributeFpnProposals', ['rois'], output_names, arg=[utils.MakeArgument('roi_canonical_scale', roi_canonical_scale), utils.MakeArgument('roi_canonical_level', roi_canonical_level), utils.MakeArgument('roi_max_level', roi_min_level + roi_num_levels - 1), utils.MakeArgument('roi_min_level', roi_min_level)], device_option=gc)
        distribute_args = {'roi_min_level': roi_min_level, 'roi_num_levels': roi_num_levels, 'roi_canonical_scale': roi_canonical_scale, 'roi_canonical_level': roi_canonical_level}
        self.assertReferenceChecks(device_option=gc, op=distribute_op, inputs=[rois, distribute_args], reference=distribute_fpn_ref)
if __name__ == '__main__':
    unittest.main()
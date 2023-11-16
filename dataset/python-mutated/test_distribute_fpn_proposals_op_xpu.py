import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types

def distribute_fpn_proposals_wrapper(fpn_rois, rois_num, min_level, max_level, refer_level, refer_scale, pixel_offset):
    if False:
        i = 10
        return i + 15
    return paddle.vision.ops.distribute_fpn_proposals(fpn_rois=fpn_rois, min_level=min_level, max_level=max_level, refer_level=refer_level, refer_scale=refer_scale, rois_num=rois_num)

class XPUTestDistributeFPNProposalsOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'distribute_fpn_proposals'
        self.use_dynamic_create_class = False

    class TestDistributeFPNProposalsOp(XPUOpTest):

        def setUp(self):
            if False:
                return 10
            self.op_type = 'distribute_fpn_proposals'
            self.python_api = distribute_fpn_proposals_wrapper
            self.python_out_sig = ['MultiFpnRois', 'RestoreIndex']
            self.dtype = self.in_type
            self.init_test_case()
            self.make_rois()
            (self.rois_fpn, self.rois_idx_restore) = self.calc_rois_distribute()
            self.inputs = {'FpnRois': (self.rois[:, 1:5], self.rois_lod)}
            self.attrs = {'max_level': self.roi_max_level, 'min_level': self.roi_min_level, 'refer_scale': self.canonical_scale, 'refer_level': self.canonical_level, 'pixel_offset': self.pixel_offset}
            output = [('out%d' % i, self.rois_fpn[i]) for i in range(len(self.rois_fpn))]
            self.outputs = {'MultiFpnRois': output, 'RestoreIndex': self.rois_idx_restore.reshape(-1, 1)}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(paddle.XPUPlace(0))

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.roi_max_level = 5
            self.roi_min_level = 2
            self.canonical_scale = 224
            self.canonical_level = 4
            self.images_shape = [512, 512]
            self.pixel_offset = True

        def boxes_area(self, boxes):
            if False:
                return 10
            offset = 1 if self.pixel_offset else 0
            w = boxes[:, 2] - boxes[:, 0] + offset
            h = boxes[:, 3] - boxes[:, 1] + offset
            areas = w * h
            assert np.all(areas >= 0), 'Negative areas founds'
            return areas

        def map_rois_to_fpn_levels(self, rois, lvl_min, lvl_max):
            if False:
                print('Hello World!')
            s = np.sqrt(self.boxes_area(rois))
            s0 = self.canonical_scale
            lvl0 = self.canonical_level
            target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-08))
            target_lvls = np.clip(target_lvls, lvl_min, lvl_max)
            return target_lvls

        def get_sub_lod(self, sub_lvl):
            if False:
                return 10
            sub_lod = [0, 0]
            max_batch_id = sub_lvl[-1]
            for i in range(max_batch_id.astype(np.int32) + 1):
                sub_lod[i] = np.where(sub_lvl == i)[0].size
            return sub_lod

        def add_multilevel_roi(self, rois, target_lvls, lvl_min, lvl_max):
            if False:
                for i in range(10):
                    print('nop')
            rois_idx_order = np.empty((0,))
            rois_fpn = []
            for lvl in range(lvl_min, lvl_max + 1):
                idx_lvl = np.where(target_lvls == lvl)[0]
                if len(idx_lvl) == 0:
                    rois_fpn.append((np.empty(shape=(0, 4)), [[0, 0]]))
                    continue
                sub_lod = self.get_sub_lod(rois[idx_lvl, 0])
                rois_fpn.append((rois[idx_lvl, 1:], [sub_lod]))
                rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
            rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
            return (rois_fpn, rois_idx_restore)

        def calc_rois_distribute(self):
            if False:
                while True:
                    i = 10
            lvl_min = self.roi_min_level
            lvl_max = self.roi_max_level
            target_lvls = self.map_rois_to_fpn_levels(self.rois[:, 1:5], lvl_min, lvl_max)
            (rois_fpn, rois_idx_restore) = self.add_multilevel_roi(self.rois, target_lvls, lvl_min, lvl_max)
            return (rois_fpn, rois_idx_restore)

        def make_rois(self):
            if False:
                for i in range(10):
                    print('nop')
            self.rois_lod = [[10, 4]]
            rois = []
            lod = self.rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (self.images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            self.rois = np.array(rois).astype('float32')

    class TestDistributeFPNProposalsOpWithRoisNum(TestDistributeFPNProposalsOp):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'distribute_fpn_proposals'
            self.python_api = distribute_fpn_proposals_wrapper
            self.python_out_sig = ['MultiFpnRois', 'MultiLevelRoIsNum', 'RestoreIndex']
            self.dtype = self.in_type
            self.init_test_case()
            self.make_rois()
            (self.rois_fpn, self.rois_idx_restore) = self.calc_rois_distribute()
            self.inputs = {'FpnRois': (self.rois[:, 1:5], self.rois_lod), 'RoisNum': np.array(self.rois_lod[0]).astype('int32')}
            self.attrs = {'max_level': self.roi_max_level, 'min_level': self.roi_min_level, 'refer_scale': self.canonical_scale, 'refer_level': self.canonical_level, 'pixel_offset': self.pixel_offset}
            output = [('out%d' % i, self.rois_fpn[i]) for i in range(len(self.rois_fpn))]
            rois_num_per_level = [('rois_num%d' % i, np.array(self.rois_fpn[i][1][0]).astype('int32')) for i in range(len(self.rois_fpn))]
            self.outputs = {'MultiFpnRois': output, 'RestoreIndex': self.rois_idx_restore.reshape(-1, 1), 'MultiLevelRoIsNum': rois_num_per_level}

    class TestDistributeFPNProposalsOpNoOffset(TestDistributeFPNProposalsOpWithRoisNum):

        def init_test_case(self):
            if False:
                while True:
                    i = 10
            self.roi_max_level = 5
            self.roi_min_level = 2
            self.canonical_scale = 224
            self.canonical_level = 4
            self.images_shape = [512, 512]
            self.pixel_offset = False
support_types = get_xpu_op_support_types('distribute_fpn_proposals')
for stype in support_types:
    create_test_class(globals(), XPUTestDistributeFPNProposalsOp, stype)
if __name__ == '__main__':
    unittest.main()
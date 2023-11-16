from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import copy

class RoIAlignRotatedOp(hu.HypothesisTestCase):

    def bbox_xywh_to_xyxy(self, boxes):
        if False:
            return 10
        '\n        Convert from [center_x center_y w h] format to [x1 y1 x2 y2].\n        '
        (w, h) = (boxes[:, 2], boxes[:, 3])
        boxes[:, 0] -= w / 2.0
        boxes[:, 1] -= h / 2.0
        boxes[:, 2] = boxes[:, 0] + w
        boxes[:, 3] = boxes[:, 1] + h
        return boxes

    @given(H=st.integers(min_value=50, max_value=100), W=st.integers(min_value=50, max_value=100), C=st.integers(min_value=1, max_value=3), num_rois=st.integers(min_value=0, max_value=10), pooled_size=st.sampled_from([7, 14]), **hu.gcs)
    def test_horizontal_rois(self, H, W, C, num_rois, pooled_size, gc, dc):
        if False:
            i = 10
            return i + 15
        '\n        Test that results match with RoIAlign when angle=0.\n        '
        X = np.random.randn(1, C, H, W).astype(np.float32)
        R = np.zeros((num_rois, 6)).astype(np.float32)
        angle = 0.0
        for i in range(num_rois):
            x = np.random.uniform(1, W - 1)
            y = np.random.uniform(1, H - 1)
            w = np.random.uniform(1, min(x, W - x))
            h = np.random.uniform(1, min(y, H - y))
            R[i] = [0, x, y, w, h, angle]
        op = core.CreateOperator('RoIAlignRotated', ['X', 'R'], ['Y'], pooled_h=pooled_size, pooled_w=pooled_size, sampling_ratio=0)

        def roialign_ref(X, R):
            if False:
                print('Hello World!')
            R_ref = copy.deepcopy(R[:, 0:5])
            R_ref[:, 1:5] = self.bbox_xywh_to_xyxy(R_ref[:, 1:5])
            ref_op = core.CreateOperator('RoIAlign', ['X_ref', 'R_ref'], ['Y_ref'], pooled_h=pooled_size, pooled_w=pooled_size, sampling_ratio=0)
            workspace.FeedBlob('X_ref', X)
            workspace.FeedBlob('R_ref', R_ref)
            workspace.RunOperatorOnce(ref_op)
            return [workspace.FetchBlob('Y_ref')]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, R], reference=roialign_ref)
        if core.IsGPUDeviceType(gc.device_type):
            self.assertGradientChecks(gc, op, [X, R], 0, [0])

    @given(H=st.integers(min_value=50, max_value=100), W=st.integers(min_value=50, max_value=100), C=st.integers(min_value=1, max_value=3), num_rois=st.integers(min_value=0, max_value=10), pooled_size=st.sampled_from([7, 14]), angle=st.sampled_from([-270, -180, -90, 90, 180, 270]), **hu.gcs)
    def test_simple_rotations(self, H, W, C, num_rois, pooled_size, angle, gc, dc):
        if False:
            i = 10
            return i + 15
        "\n        Test with right-angled rotations that don't need interpolation.\n        "
        X = np.random.randn(1, C, H, W).astype(np.float32)
        R = np.zeros((num_rois, 6)).astype(np.float32)
        for i in range(num_rois):
            x = np.random.uniform(1, W - 1)
            y = np.random.uniform(1, H - 1)
            w = np.random.uniform(1, min(x, W - x, y, H - y))
            h = np.random.uniform(1, min(x, W - x, y, H - y))
            R[i] = [0, x, y, w, h, angle]
        op = core.CreateOperator('RoIAlignRotated', ['X', 'R'], ['Y'], pooled_h=pooled_size, pooled_w=pooled_size, sampling_ratio=0)

        def roialign_rot90(m, k=1, axes=(0, 1)):
            if False:
                print('Hello World!')
            axes = tuple(axes)
            if len(axes) != 2:
                raise ValueError('len(axes) must be 2.')
            m = np.asanyarray(m)
            if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
                raise ValueError('Axes must be different.')
            if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or (axes[1] < -m.ndim):
                raise ValueError('Axes={} out of range for array of ndim={}.'.format(axes, m.ndim))
            k %= 4
            if k == 0:
                return m[:]
            if k == 2:
                return roialign_flip(roialign_flip(m, axes[0]), axes[1])
            axes_list = np.arange(0, m.ndim)
            (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])
            if k == 1:
                return np.transpose(roialign_flip(m, axes[1]), axes_list)
            else:
                return roialign_flip(np.transpose(m, axes_list), axes[1])

        def roialign_flip(m, axis):
            if False:
                print('Hello World!')
            if not hasattr(m, 'ndim'):
                m = np.asarray(m)
            indexer = [slice(None)] * m.ndim
            try:
                indexer[axis] = slice(None, None, -1)
            except IndexError as e:
                raise ValueError('axis=%i is invalid for the %i-dimensional input array' % (axis, m.ndim)) from e
            return m[tuple(indexer)]

        def roialign_ref(X, R):
            if False:
                return 10
            norm_angle = (angle + 360) % 360
            X_ref = roialign_rot90(X, k=-norm_angle / 90, axes=(2, 3))
            (roi_x, roi_y) = (R[:, 1], R[:, 2])
            if norm_angle == 90:
                new_roi_x = H - roi_y - 1
                new_roi_y = roi_x
            elif norm_angle == 180:
                new_roi_x = W - roi_x - 1
                new_roi_y = H - roi_y - 1
            elif norm_angle == 270:
                new_roi_x = roi_y
                new_roi_y = W - roi_x - 1
            else:
                raise NotImplementedError
            R_ref = copy.deepcopy(R[:, 0:5])
            (R_ref[:, 1], R_ref[:, 2]) = (new_roi_x, new_roi_y)
            R_ref[:, 1:5] = self.bbox_xywh_to_xyxy(R_ref[:, 1:5])
            ref_op = core.CreateOperator('RoIAlign', ['X_ref', 'R_ref'], ['Y_ref'], pooled_h=pooled_size, pooled_w=pooled_size, sampling_ratio=0)
            workspace.FeedBlob('X_ref', X_ref)
            workspace.FeedBlob('R_ref', R_ref)
            workspace.RunOperatorOnce(ref_op)
            return [workspace.FetchBlob('Y_ref')]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, R], reference=roialign_ref)
        if core.IsGPUDeviceType(gc.device_type):
            self.assertGradientChecks(gc, op, [X, R], 0, [0])
if __name__ == '__main__':
    import unittest
    unittest.main()
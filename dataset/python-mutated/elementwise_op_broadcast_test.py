import unittest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import numpy as np
import operator
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

class TestElementwiseBroadcast(serial.SerializedTestCase):

    def __generate_test_cases(self, allow_broadcast_fastpath: bool):
        if False:
            while True:
                i = 10
        '\n        generates a set of test cases\n\n        For each iteration, generates X, Y, args, X_out, Y_out\n        where\n          X, Y         are test input tensors\n          args         is a dictionary of arguments to be passed to\n                       core.CreateOperator()\n          X_out, Y_out are reshaped versions of X and Y\n                       which can be used to calculate the expected\n                       result with the operator to be tested\n        '
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(4, 5).astype(np.float32)
        args = dict(broadcast=1, allow_broadcast_fastpath=allow_broadcast_fastpath)
        yield (X, Y, args, X, Y)
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        args = dict(broadcast=1, axis=1, allow_broadcast_fastpath=allow_broadcast_fastpath)
        yield (X, Y, args, X, Y[:, :, np.newaxis])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(2).astype(np.float32)
        args = dict(broadcast=1, axis=0, allow_broadcast_fastpath=allow_broadcast_fastpath)
        yield (X, Y, args, X, Y[:, np.newaxis, np.newaxis, np.newaxis])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(1, 4, 1).astype(np.float32)
        args = dict(broadcast=1, axis=1, allow_broadcast_fastpath=allow_broadcast_fastpath)
        yield (X, Y, args, X, Y)

    def __test_binary_op(self, gc, dc, caffe2_op, op_function, allow_broadcast_fastpath: bool=False):
        if False:
            return 10
        '\n        Args:\n            caffe2_op: A string. Name of the caffe operator to test.\n            op_function: an actual python operator (e.g. operator.add)\n        path_prefix: A string. Optional param used to construct db name or path\n            where checkpoint files are stored.\n        '
        for (X, Y, op_args, X_out, Y_out) in self.__generate_test_cases(allow_broadcast_fastpath):
            op = core.CreateOperator(caffe2_op, ['X', 'Y'], 'out', **op_args)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('Y', Y)
            workspace.RunOperatorOnce(op)
            out = workspace.FetchBlob('out')
            np.testing.assert_array_almost_equal(out, op_function(X_out, Y_out))
            self.assertDeviceChecks(dc, op, [X, Y], [0])
            self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(allow_broadcast_fastpath=st.booleans(), **hu.gcs)
    @settings(deadline=None)
    def test_broadcast_Add(self, allow_broadcast_fastpath: bool, gc, dc):
        if False:
            i = 10
            return i + 15
        self.__test_binary_op(gc, dc, 'Add', operator.add, allow_broadcast_fastpath=allow_broadcast_fastpath)

    @given(allow_broadcast_fastpath=st.booleans(), **hu.gcs)
    @settings(deadline=None)
    def test_broadcast_Mul(self, allow_broadcast_fastpath: bool, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        self.__test_binary_op(gc, dc, 'Mul', operator.mul, allow_broadcast_fastpath=allow_broadcast_fastpath)

    @given(allow_broadcast_fastpath=st.booleans(), **hu.gcs)
    @settings(deadline=None)
    def test_broadcast_Sub(self, allow_broadcast_fastpath: bool, gc, dc):
        if False:
            print('Hello World!')
        self.__test_binary_op(gc, dc, 'Sub', operator.sub, allow_broadcast_fastpath=allow_broadcast_fastpath)

    @given(**hu.gcs)
    @settings(deadline=None)
    def test_broadcast_powt(self, gc, dc):
        if False:
            print('Hello World!')
        np.random.seed(101)

        def powt_op(X, Y):
            if False:
                print('Hello World!')
            return [np.power(X, Y)]

        def powt_grad(g_out, outputs, fwd_inputs):
            if False:
                print('Hello World!')
            [X, Y] = fwd_inputs
            Z = outputs[0]
            return [Y * np.power(X, Y - 1), Z * np.log(X)] * g_out
        X = np.random.rand(2, 3, 4, 5).astype(np.float32) + 1.0
        Y = np.random.rand(4, 5).astype(np.float32) + 2.0

        def powt_grad_broadcast(g_out, outputs, fwd_inputs):
            if False:
                i = 10
                return i + 15
            [GX, GY] = powt_grad(g_out, outputs, fwd_inputs)
            return [GX, np.sum(np.sum(GY, 1), 0)]
        op = core.CreateOperator('Pow', ['X', 'Y'], 'Z', broadcast=1)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=powt_op, output_to_grad='Z', grad_reference=powt_grad_broadcast)
        X = np.random.rand(2, 3, 4, 5).astype(np.float32) + 1.0
        Y = np.random.rand(3, 4).astype(np.float32) + 2.0

        def powt_op_axis1(X, Y):
            if False:
                i = 10
                return i + 15
            return powt_op(X, Y[:, :, np.newaxis])

        def powt_grad_axis1(g_out, outputs, fwd_inputs):
            if False:
                return 10
            [X, Y] = fwd_inputs
            [GX, GY] = powt_grad(g_out, outputs, [X, Y[:, :, np.newaxis]])
            return [GX, np.sum(np.sum(GY, 3), 0)]
        op = core.CreateOperator('Pow', ['X', 'Y'], 'Z', broadcast=1, axis=1)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=powt_op_axis1, output_to_grad='Z', grad_reference=powt_grad_axis1)
        X = np.random.rand(2, 3, 4, 5).astype(np.float32) + 1.0
        Y = np.random.rand(2).astype(np.float32) + 2.0

        def powt_op_axis0(X, Y):
            if False:
                return 10
            return powt_op(X, Y[:, np.newaxis, np.newaxis, np.newaxis])

        def powt_grad_axis0(g_out, outputs, fwd_inputs):
            if False:
                i = 10
                return i + 15
            [X, Y] = fwd_inputs
            [GX, GY] = powt_grad(g_out, outputs, [X, Y[:, np.newaxis, np.newaxis, np.newaxis]])
            return [GX, np.sum(np.sum(np.sum(GY, 3), 2), 1)]
        op = core.CreateOperator('Pow', ['X', 'Y'], 'Z', broadcast=1, axis=0)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=powt_op_axis0, output_to_grad='Z', grad_reference=powt_grad_axis0)
        X = np.random.rand(2, 3, 4, 5).astype(np.float32) + 1.0
        Y = np.random.rand(1, 4, 1).astype(np.float32) + 2.0

        def powt_op_mixed(X, Y):
            if False:
                for i in range(10):
                    print('nop')
            return powt_op(X, Y[np.newaxis, :, :, :])

        def powt_grad_mixed(g_out, outputs, fwd_inputs):
            if False:
                for i in range(10):
                    print('nop')
            [X, Y] = fwd_inputs
            [GX, GY] = powt_grad(g_out, outputs, [X, Y[np.newaxis, :, :, :]])
            return [GX, np.reshape(np.sum(np.sum(np.sum(GY, 3), 1), 0), (1, 4, 1))]
        op = core.CreateOperator('Pow', ['X', 'Y'], 'Z', broadcast=1, axis=1)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=powt_op_mixed, output_to_grad='Z', grad_reference=powt_grad_mixed)

    @given(allow_broadcast_fastpath=st.booleans(), **hu.gcs)
    def test_broadcast_scalar(self, allow_broadcast_fastpath: bool, gc, dc):
        if False:
            return 10
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(1).astype(np.float32)
        op = core.CreateOperator('Add', ['X', 'Y'], 'out', broadcast=1, allow_broadcast_fastpath=allow_broadcast_fastpath)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        np.testing.assert_array_almost_equal(out, X + Y)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        X = np.random.rand(1).astype(np.float32)
        Y = np.random.rand(1).astype(np.float32).reshape([])
        op = core.CreateOperator('Add', ['X', 'Y'], 'out', broadcast=1, allow_broadcast_fastpath=allow_broadcast_fastpath)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        np.testing.assert_array_almost_equal(out, X + Y)
        self.assertDeviceChecks(dc, op, [X, Y], [0])

    @given(allow_broadcast_fastpath=st.booleans(), **hu.gcs)
    def test_semantic_broadcast(self, allow_broadcast_fastpath: bool, gc, dc):
        if False:
            return 10
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(3).astype(np.float32)
        op = core.CreateOperator('Add', ['X', 'Y'], 'out', broadcast=1, axis_str='C', allow_broadcast_fastpath=allow_broadcast_fastpath)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        np.testing.assert_array_almost_equal(out, X + Y[:, np.newaxis, np.newaxis])
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(5).astype(np.float32)
        op = core.CreateOperator('Add', ['X', 'Y'], 'out', broadcast=1, axis_str='C', order='NHWC', allow_broadcast_fastpath=allow_broadcast_fastpath)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        np.testing.assert_array_almost_equal(out, X + Y)
        self.assertDeviceChecks(dc, op, [X, Y], [0])

    @given(**hu.gcs)
    def test_sum_reduce_empty_blob(self, gc, dc):
        if False:
            while True:
                i = 10
        net = core.Net('test')
        with core.DeviceScope(gc):
            net.GivenTensorFill([], ['X'], values=[], shape=[2, 0, 5])
            net.GivenTensorFill([], ['Y'], values=[], shape=[2, 0])
            net.SumReduceLike(['X', 'Y'], 'out', axis=0)
            workspace.RunNetOnce(net)

    @given(**hu.gcs)
    def test_sum_reduce(self, gc, dc):
        if False:
            while True:
                i = 10
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(4, 5).astype(np.float32)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        res = np.sum(X, axis=0)
        res = np.sum(res, axis=0)
        np.testing.assert_array_almost_equal(out, res)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(2, 3).astype(np.float32)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1, axis=0)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        res = np.sum(X, axis=3)
        res = np.sum(res, axis=2)
        np.testing.assert_array_almost_equal(out, res, decimal=3)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(3, 4).astype(np.float32)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1, axis=1)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        res = np.sum(X, axis=0)
        res = np.sum(res, axis=2)
        np.testing.assert_array_almost_equal(out, res)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        X = np.random.rand(2, 3, 4, 500).astype(np.float64)
        Y = np.random.rand(1).astype(np.float64)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        res = np.array(np.sum(X))
        np.testing.assert_array_almost_equal(out, res, decimal=0)
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        Y = np.random.rand(1, 3, 4, 1).astype(np.float32)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1)
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)
        workspace.RunOperatorOnce(op)
        out = workspace.FetchBlob('out')
        res = np.sum(X, axis=0)
        res = np.sum(res, axis=2).reshape(Y.shape)
        np.testing.assert_array_almost_equal(out, res)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        dc_cpu_only = [d for d in dc if d.device_type != caffe2_pb2.CUDA]
        self.assertDeviceChecks(dc_cpu_only, op, [X, Y], [0])

    @unittest.skipIf(not workspace.has_gpu_support, 'No gpu support')
    @given(**hu.gcs)
    def test_sum_reduce_fp16(self, gc, dc):
        if False:
            while True:
                i = 10
        assume(core.IsGPUDeviceType(gc.device_type))
        X = np.random.rand(2, 3, 4, 5).astype(np.float16)
        Y = np.random.rand(4, 5).astype(np.float16)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1, device_option=gc)

        def ref_op(X, Y):
            if False:
                i = 10
                return i + 15
            res = np.sum(X, axis=0)
            res = np.sum(res, axis=0)
            return [res]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=ref_op, threshold=0.001)
        X = np.random.rand(2, 3, 4, 5).astype(np.float16)
        Y = np.random.rand(2, 3).astype(np.float16)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1, axis=0)

        def ref_op(X, Y):
            if False:
                i = 10
                return i + 15
            res = np.sum(X, axis=3)
            res = np.sum(res, axis=2)
            return [res]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=ref_op, threshold=0.001)
        X = np.random.rand(2, 3, 4, 5).astype(np.float16)
        Y = np.random.rand(3, 4).astype(np.float16)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1, axis=1)

        def ref_op(X, Y):
            if False:
                for i in range(10):
                    print('nop')
            res = np.sum(X, axis=0)
            res = np.sum(res, axis=2)
            return [res]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=ref_op, threshold=0.001)
        X = np.random.rand(2, 3, 4, 5).astype(np.float16)
        Y = np.random.rand(1, 3, 4, 1).astype(np.float16)
        op = core.CreateOperator('SumReduceLike', ['X', 'Y'], 'out', broadcast=1)

        def ref_op(X, Y):
            if False:
                while True:
                    i = 10
            res = np.sum(X, axis=0)
            res = np.sum(res, axis=2)
            return [res.reshape(Y.shape)]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X, Y], reference=ref_op, threshold=0.001)
if __name__ == '__main__':
    unittest.main()
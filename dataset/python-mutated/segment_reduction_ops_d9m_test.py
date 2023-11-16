"""Tests for deterministic functionality of segment reduction ops."""
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class SegmentReductionDeterminismExceptionsTest(test.TestCase):
    """Test d9m-unimplemented exceptions from the segment reduction ops.

  Test that tf.errors.UnimplementedError is thrown or not thrown, as
  appropriate, by the GPU code-paths for segment reduction ops when
  deterministic ops are enabled.

  This test assumes that the base op test runs all the same test cases when
  deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

    def _input(self, data_type, segment_ids_type):
        if False:
            return 10
        data = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=data_type)
        segment_ids = constant_op.constant([0, 1], dtype=segment_ids_type)
        num_segments = 2
        return (data, segment_ids, num_segments)

    @test_util.run_cuda_only
    def testSortedOps(self):
        if False:
            for i in range(10):
                print('nop')
        op_should_throw_for_float = {math_ops.segment_max: False, math_ops.segment_min: False, math_ops.segment_prod: True, math_ops.segment_sum: True}
        for (op, should_throw_for_float) in op_should_throw_for_float.items():
            for segment_ids_type in [dtypes.int32, dtypes.int64]:
                for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
                    with self.cached_session(force_gpu=True):
                        (data, segment_ids, _) = self._input(data_type, segment_ids_type)
                        result = op(data, segment_ids)
                        self.evaluate(result)
    _UNSORTED_ERROR_MESSAGE = 'Deterministic GPU implementation of unsorted ' + 'segment reduction op not available.'

    @test_util.run_cuda_only
    @test_util.run_in_graph_and_eager_modes
    def testUnsortedOps(self):
        if False:
            while True:
                i = 10
        op_should_throw_for_float = {math_ops.unsorted_segment_max: False, math_ops.unsorted_segment_min: False, math_ops.unsorted_segment_mean: True, math_ops.unsorted_segment_sqrt_n: True, math_ops.unsorted_segment_prod: True, math_ops.unsorted_segment_sum: True}
        with self.session(force_gpu=True):
            for (op, should_throw_for_float) in op_should_throw_for_float.items():
                for segment_ids_type in [dtypes.int32, dtypes.int64]:
                    for data_type in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32]:
                        if op == math_ops.unsorted_segment_sqrt_n and data_type == dtypes.int32:
                            continue
                        (data, segment_ids, num_segments) = self._input(data_type, segment_ids_type)
                        result = op(data, segment_ids, num_segments)
                        self.evaluate(result)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='No ROCm support for complex types in segment reduction ops')
    @test_util.run_cuda_only
    def testUnsortedOpsComplex(self):
        if False:
            i = 10
            return i + 15
        for op in [math_ops.unsorted_segment_sum]:
            for data_type in [dtypes.complex64, dtypes.complex128]:
                for segment_ids_type in [dtypes.int32, dtypes.int64]:
                    with self.cached_session(force_gpu=True):
                        (data, segment_ids, num_segments) = self._input(data_type, segment_ids_type)
                        result = op(data, segment_ids, num_segments)
                        self.evaluate(result)

    @test_util.run_cuda_only
    @test_util.run_in_graph_and_eager_modes
    def testConvertToTensor(self):
        if False:
            print('Hello World!')
        with self.session(force_gpu=True):
            dtypes_to_test = [dtypes.float16, dtypes.float32, dtypes.float64]
            if not test.is_built_with_rocm():
                dtypes_to_test += [dtypes.complex64, dtypes.complex128]
            for data_type in dtypes_to_test:
                for segment_ids_type in [dtypes.int32, dtypes.int64]:
                    (values, indices, _) = self._input(data_type, segment_ids_type)
                    sparse_value = indexed_slices.IndexedSlices(values, indices, dense_shape=values.shape)
                    result = ops.convert_to_tensor(sparse_value)
                    self.evaluate(result)

    @test_util.run_cuda_only
    def testGatherBackprop(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = [dtypes.float16, dtypes.float32, dtypes.float64]
        if not test.is_built_with_rocm():
            dtypes_to_test += [dtypes.complex64, dtypes.complex128]
        for data_type in dtypes_to_test:
            for segment_ids_type in [dtypes.int32, dtypes.int64]:
                with self.cached_session(force_gpu=True):
                    (params, indices, _) = self._input(data_type, segment_ids_type)
                    params = variables.Variable(params)
                    with backprop.GradientTape() as tape:
                        tape.watch(params)
                        op_output = array_ops.gather(params, indices)
                    gradient = tape.gradient(op_output, params)
                    self.evaluate(params.assign(gradient))
if __name__ == '__main__':
    config.enable_op_determinism()
    test.main()
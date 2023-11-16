from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

class TestLengthSplitOperator(serial.SerializedTestCase):

    def _length_split_op_ref(self, input_lengths, n_split_array):
        if False:
            while True:
                i = 10
        output = []
        n_split = n_split_array[0]
        for x in input_lengths:
            mod = x % n_split
            val = x // n_split + 1
            for _ in range(n_split):
                if mod > 0:
                    output.append(val)
                    mod -= 1
                else:
                    output.append(val - 1)
        return [np.array(output).astype(np.int32)]

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_length_split_edge(self, gc, dc):
        if False:
            while True:
                i = 10
        input_lengths = np.array([3, 4, 5]).astype(np.int32)
        n_split_ = np.array([5]).astype(np.int32)
        op = core.CreateOperator('LengthsSplit', ['input_lengths', 'n_split'], ['Y'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[input_lengths, n_split_], reference=self._length_split_op_ref)
        self.assertDeviceChecks(dc, op, [input_lengths, n_split_], [0])

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_length_split_arg(self, gc, dc):
        if False:
            while True:
                i = 10
        input_lengths = np.array([9, 4, 5]).astype(np.int32)
        n_split = 3
        op = core.CreateOperator('LengthsSplit', ['input_lengths'], ['Y'], n_split=n_split)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[input_lengths], reference=lambda x: self._length_split_op_ref(x, [n_split]))
        self.assertDeviceChecks(dc, op, [input_lengths], [0])

    @given(**hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_length_split_override_arg(self, gc, dc):
        if False:
            while True:
                i = 10
        input_lengths = np.array([9, 4, 5]).astype(np.int32)
        n_split_ignored = 2
        n_split_used = np.array([3]).astype(np.int32)
        op = core.CreateOperator('LengthsSplit', ['input_lengths', 'n_split'], ['Y'], n_split=n_split_ignored)
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[input_lengths, n_split_used], reference=self._length_split_op_ref)
        self.assertDeviceChecks(dc, op, [input_lengths, n_split_used], [0])

    @given(m=st.integers(1, 100), n_split=st.integers(1, 20), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_length_split_even_divide(self, m, n_split, gc, dc):
        if False:
            i = 10
            return i + 15
        input_lengths = np.random.randint(100, size=m).astype(np.int32) * n_split
        n_split_ = np.array([n_split]).astype(np.int32)
        op = core.CreateOperator('LengthsSplit', ['input_lengths', 'n_split'], ['Y'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[input_lengths, n_split_], reference=self._length_split_op_ref)
        self.assertDeviceChecks(dc, op, [input_lengths, n_split_], [0])

    @given(m=st.integers(1, 100), n_split=st.integers(1, 20), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_length_split_random(self, m, n_split, gc, dc):
        if False:
            while True:
                i = 10
        input_lengths = np.random.randint(100, size=m).astype(np.int32)
        n_split_ = np.array([n_split]).astype(np.int32)
        op = core.CreateOperator('LengthsSplit', ['input_lengths', 'n_split'], ['Y'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[input_lengths, n_split_], reference=self._length_split_op_ref)
        self.assertDeviceChecks(dc, op, [input_lengths, n_split_], [0])
if __name__ == '__main__':
    import unittest
    unittest.main()
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

class TestPackRNNSequenceOperator(serial.SerializedTestCase):

    @serial.given(n=st.integers(0, 10), k=st.integers(1, 5), dim=st.integers(1, 5), **hu.gcs_cpu_only)
    def test_pack_rnn_seqence(self, n, k, dim, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        values = np.random.rand(sum(lengths), dim).astype(np.float32)

        def pack_op(values, lengths):
            if False:
                return 10
            T = max(lengths) if any(lengths) else 0
            N = lengths.size
            output = np.zeros((T, N) + values.shape[1:]).astype(np.float32)
            offset = 0
            for c in range(N):
                for r in range(lengths[c]):
                    output[r][c] = values[offset + r]
                offset += lengths[c]
            return [output]
        op = core.CreateOperator('PackRNNSequence', ['values', 'lengths'], 'out')
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[values, lengths], reference=pack_op)
        self.assertDeviceChecks(dc, op, [values, lengths], [0])
        self.assertGradientChecks(gc, op, [values, lengths], 0, [0])

    @serial.given(n=st.integers(0, 10), k=st.integers(2, 5), dim=st.integers(1, 5), **hu.gcs_cpu_only)
    def test_unpack_rnn_seqence(self, n, k, dim, gc, dc):
        if False:
            print('Hello World!')
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        T = max(lengths) if any(lengths) else 0
        N = lengths.size
        values = np.random.rand(T, N, dim).astype(np.float32)

        def unpack_op(values, lengths):
            if False:
                return 10
            M = sum(lengths)
            output = np.zeros((M,) + values.shape[2:]).astype(np.float32)
            N = lengths.size
            offset = 0
            for c in range(N):
                for r in range(lengths[c]):
                    output[offset + r] = values[r][c]
                offset += lengths[c]
            return [output]
        op = core.CreateOperator('UnpackRNNSequence', ['values', 'lengths'], 'out')
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[values, lengths], reference=unpack_op)
        self.assertDeviceChecks(dc, op, [values, lengths], [0])
        self.assertGradientChecks(gc, op, [values, lengths], 0, [0])
if __name__ == '__main__':
    import unittest
    unittest.main()
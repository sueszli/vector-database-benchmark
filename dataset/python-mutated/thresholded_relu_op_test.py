from caffe2.python import core
from hypothesis import given, settings
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import numpy as np
import unittest

class TestThresholdedRelu(serial.SerializedTestCase):

    @serial.given(input=hu.tensor(), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_thresholded_relu_1(self, input, gc, dc, engine):
        if False:
            print('Hello World!')
        X = input
        op = core.CreateOperator('ThresholdedRelu', ['X'], ['Y'], engine=engine)

        def defaultRef(X):
            if False:
                i = 10
                return i + 15
            Y = np.copy(X)
            Y[Y <= 1.0] = 0.0
            return (Y,)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], defaultRef)

    @given(input=hu.tensor(), alpha=st.floats(min_value=1.0, max_value=5.0), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_thresholded_relu_2(self, input, alpha, gc, dc, engine):
        if False:
            for i in range(10):
                print('nop')
        X = input
        op = core.CreateOperator('ThresholdedRelu', ['X'], ['Y'], alpha=alpha, engine=engine)

        def ref(X):
            if False:
                return 10
            Y = np.copy(X)
            Y[Y <= alpha] = 0.0
            return (Y,)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], ref)

    @given(input=hu.tensor(), alpha=st.floats(min_value=1.1, max_value=5.0), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    @settings(deadline=10000)
    def test_thresholded_relu_3(self, input, alpha, gc, dc, engine):
        if False:
            while True:
                i = 10
        X = TestThresholdedRelu.fix_input(input)
        op = core.CreateOperator('ThresholdedRelu', ['X'], ['Y'], alpha=float(alpha), engine=engine)
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @staticmethod
    def fix_input(input):
        if False:
            i = 10
            return i + 15
        input += 0.02 * np.sign(input)
        return input
if __name__ == '__main__':
    unittest.main()
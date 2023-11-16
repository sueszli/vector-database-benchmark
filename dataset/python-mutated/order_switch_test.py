import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, utils
from hypothesis import given, settings

class OrderSwitchOpsTest(hu.HypothesisTestCase):

    @given(X=hu.tensor(min_dim=3, max_dim=5, min_value=1, max_value=5), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    @settings(deadline=10000)
    def test_nchw2nhwc(self, X, engine, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        op = core.CreateOperator('NCHW2NHWC', ['X'], ['Y'], engine=engine)

        def nchw2nhwc_ref(X):
            if False:
                while True:
                    i = 10
            return (utils.NCHW2NHWC(X),)
        self.assertReferenceChecks(gc, op, [X], nchw2nhwc_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(min_dim=3, max_dim=5, min_value=1, max_value=5), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    @settings(deadline=10000)
    def test_nhwc2nchw(self, X, engine, gc, dc):
        if False:
            print('Hello World!')
        op = core.CreateOperator('NHWC2NCHW', ['X'], ['Y'], engine=engine)

        def nhwc2nchw_ref(X):
            if False:
                while True:
                    i = 10
            return (utils.NHWC2NCHW(X),)
        self.assertReferenceChecks(gc, op, [X], nhwc2nchw_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
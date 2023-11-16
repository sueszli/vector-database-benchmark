from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
from functools import partial
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

def _unique_ref(x, return_inverse):
    if False:
        return 10
    ret = np.unique(x, return_inverse=return_inverse)
    if not return_inverse:
        ret = [ret]
    return ret

class TestUniqueOps(serial.SerializedTestCase):

    @given(X=hu.tensor1d(min_len=0, dtype=np.int32, elements=st.integers(min_value=-10, max_value=10)), return_remapping=st.booleans(), **hu.gcs_no_hip)
    @settings(deadline=10000)
    def test_unique_op(self, X, return_remapping, gc, dc):
        if False:
            print('Hello World!')
        X = np.sort(X)
        op = core.CreateOperator('Unique', ['X'], ['U', 'remap'] if return_remapping else ['U'])
        self.assertDeviceChecks(device_options=dc, op=op, inputs=[X], outputs_to_check=[0, 1] if return_remapping else [0])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X], reference=partial(_unique_ref, return_inverse=return_remapping))
if __name__ == '__main__':
    import unittest
    unittest.main()
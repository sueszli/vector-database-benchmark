from hypothesis import given
import numpy as np
import hypothesis.strategies as st
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

@st.composite
def _dev_options(draw):
    if False:
        return 10
    op_dev = draw(st.sampled_from(hu.device_options))
    if op_dev == hu.cpu_do:
        input_blob_dev = hu.cpu_do
    else:
        input_blob_dev = draw(st.sampled_from(hu.device_options))
    return (op_dev, input_blob_dev)

class TestEnsureCPUOutputOp(hu.HypothesisTestCase):

    @given(input=hu.tensor(dtype=np.float32), dev_options=_dev_options())
    def test_ensure_cpu_output(self, input, dev_options):
        if False:
            i = 10
            return i + 15
        (op_dev, input_blob_dev) = dev_options
        net = core.Net('test_net')
        data = net.GivenTensorFill([], ['data'], values=input, shape=input.shape, device_option=input_blob_dev)
        data_cpu = net.EnsureCPUOutput([data], ['data_cpu'], device_option=op_dev)
        workspace.RunNetOnce(net)
        data_cpu_value = workspace.FetchBlob(data_cpu)
        np.testing.assert_allclose(input, data_cpu_value)
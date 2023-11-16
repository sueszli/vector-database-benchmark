from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

def entropy(p):
    if False:
        for i in range(10):
            print('nop')
    q = 1.0 - p
    return -p * np.log(p) - q * np.log(q)

def jsd(p, q):
    if False:
        print('Hello World!')
    return [entropy(p / 2.0 + q / 2.0) - entropy(p) / 2.0 - entropy(q) / 2.0]

def jsd_grad(go, o, pq_list):
    if False:
        for i in range(10):
            print('nop')
    (p, q) = pq_list
    m = (p + q) / 2.0
    return [np.log(p * (1 - m) / (1 - p) / m) / 2.0 * go, None]

class TestJSDOps(serial.SerializedTestCase):

    @serial.given(n=st.integers(10, 100), **hu.gcs_cpu_only)
    def test_bernoulli_jsd(self, n, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        p = np.random.rand(n).astype(np.float32)
        q = np.random.rand(n).astype(np.float32)
        op = core.CreateOperator('BernoulliJSD', ['p', 'q'], ['l'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[p, q], reference=jsd, output_to_grad='l', grad_reference=jsd_grad)
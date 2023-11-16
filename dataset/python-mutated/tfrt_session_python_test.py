"""Tests for tensorflow.python.client.session.Session (with tfrt_session)."""
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class TfrtSessionPythonTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TfrtSessionPythonTest, self).setUp()
        self._config = config_pb2.ConfigProto(experimental=config_pb2.ConfigProto.Experimental(use_tfrt=True))

    def testUseExistingGraph(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g, ops.device('/cpu:0'):
            a = constant_op.constant(6.0, shape=[1, 1])
            b = constant_op.constant(7.0, shape=[1, 1])
            c = math_ops.matmul(a, b, name='matmul')
        with session.Session(graph=g, config=self._config):
            result = c.eval()
            self.assertAllEqual(result, [[42.0]])

    def testUseDefaultGraph(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default(), ops.device('/cpu:0'):
            a = constant_op.constant(6.0, shape=[1, 1])
            b = constant_op.constant(7.0, shape=[1, 1])
            c = math_ops.matmul(a, b, name='matmul')
            with session.Session(config=self._config):
                result = c.eval()
                self.assertAllEqual(result, [[42.0]])
if __name__ == '__main__':
    googletest.main()
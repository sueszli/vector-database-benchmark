import unittest
import numpy as np
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

class TestPredictor(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(1)
        self.predict_net = self._predict_net
        self.init_net = self._init_net

    @property
    def _predict_net(self):
        if False:
            return 10
        net = caffe2_pb2.NetDef()
        net.name = 'test-predict-net'
        net.external_input[:] = ['A', 'B']
        net.external_output[:] = ['C']
        net.op.extend([core.CreateOperator('MatMul', ['A', 'B'], ['C'])])
        return net.SerializeToString()

    @property
    def _init_net(self):
        if False:
            return 10
        net = caffe2_pb2.NetDef()
        net.name = 'test-init-net'
        net.external_output[:] = ['A', 'B']
        net.op.extend([core.CreateOperator('GivenTensorFill', [], ['A'], shape=(2, 3), values=np.zeros((2, 3), np.float32).flatten().tolist()), core.CreateOperator('GivenTensorFill', [], ['B'], shape=(3, 4), values=np.zeros((3, 4), np.float32).flatten().tolist())])
        return net.SerializeToString()

    def test_run(self):
        if False:
            return 10
        A = np.ones((2, 3), np.float32)
        B = np.ones((3, 4), np.float32)
        predictor = workspace.Predictor(self.init_net, self.predict_net)
        outputs = predictor.run([A, B])
        self.assertEqual(len(outputs), 1)
        np.testing.assert_almost_equal(np.dot(A, B), outputs[0])

    def test_run_map(self):
        if False:
            while True:
                i = 10
        A = np.zeros((2, 3), np.float32)
        B = np.ones((3, 4), np.float32)
        predictor = workspace.Predictor(self.init_net, self.predict_net)
        outputs = predictor.run({'B': B})
        self.assertEqual(len(outputs), 1)
        np.testing.assert_almost_equal(np.dot(A, B), outputs[0])
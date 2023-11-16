import os
import shutil
import tempfile
import unittest
import numpy as np
import paddle
from paddle.static import InputSpec
from paddle.vision import models

class TestPretrainedModel(unittest.TestCase):

    def infer(self, arch):
        if False:
            while True:
                i = 10
        path = os.path.join(tempfile.mkdtemp(), '.cache_test_pretrained_model')
        if not os.path.exists(path):
            os.makedirs(path)
        x = np.array(np.random.random((2, 3, 224, 224)), dtype=np.float32)
        res = {}
        for dygraph in [True, False]:
            if not dygraph:
                paddle.enable_static()
            net = models.__dict__[arch](pretrained=True)
            inputs = [InputSpec([None, 3, 224, 224], 'float32', 'image')]
            model = paddle.Model(network=net, inputs=inputs)
            model.prepare()
            if dygraph:
                model.save(path)
                res['dygraph'] = model.predict_batch(x)
            else:
                model.load(path)
                res['static'] = model.predict_batch(x)
            if not dygraph:
                paddle.disable_static()
        shutil.rmtree(path)
        np.testing.assert_allclose(res['dygraph'], res['static'])

    def test_models(self):
        if False:
            for i in range(10):
                print('nop')
        arches = ['mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'squeezenet1_0', 'shufflenet_v2_x0_25']
        for arch in arches:
            self.infer(arch)
if __name__ == '__main__':
    unittest.main()
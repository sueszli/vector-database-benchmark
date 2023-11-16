import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.base import core
paddle.enable_static()

class TestDistModelRun(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        print('cleaned up the env')
        self.temp_dir.cleanup()

    def test_dist_model_run(self):
        if False:
            i = 10
            return i + 15
        path_prefix = os.path.join(self.temp_dir.name, 'dist_model_run_test/inf')
        x = paddle.static.data(name='x', shape=[28, 28], dtype='float32')
        y = paddle.static.data(name='y', shape=[28, 1], dtype='int64')
        predict = paddle.static.nn.fc(x, 10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(predict, y)
        avg_loss = paddle.tensor.stat.mean(loss)
        exe = paddle.static.Executor(paddle.XPUPlace(0))
        exe.run(paddle.static.default_startup_program())
        x_data = np.random.randn(28, 28).astype('float32')
        y_data = np.random.randint(0, 9, size=[28, 1]).astype('int64')
        exe.run(paddle.static.default_main_program(), feed={'x': x_data, 'y': y_data}, fetch_list=[avg_loss])
        paddle.static.save_inference_model(path_prefix, [x, y], [avg_loss], exe)
        print('save model to', path_prefix)
        x_tensor = np.random.randn(28, 28).astype('float32')
        y_tensor = np.random.randint(0, 9, size=[28, 1]).astype('int64')
        config = core.DistModelConfig()
        config.model_dir = path_prefix
        config.place = 'XPU'
        dist = core.DistModel(config)
        dist.init()
        dist_x = core.DistModelTensor(x_tensor, 'x')
        dist_y = core.DistModelTensor(y_tensor, 'y')
        input_data = [dist_x, dist_y]
        output_rst = dist.run(input_data)
        dist_model_rst = output_rst[0].as_ndarray().ravel().tolist()
        print('dist model rst:', dist_model_rst)
        [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(path_prefix, exe)
        results = exe.run(inference_program, feed={'x': x_tensor, 'y': y_tensor}, fetch_list=fetch_targets)
        load_inference_model_rst = results[0]
        print('load inference model api rst:', load_inference_model_rst)
        np.testing.assert_allclose(dist_model_rst, load_inference_model_rst)
if __name__ == '__main__':
    unittest.main()
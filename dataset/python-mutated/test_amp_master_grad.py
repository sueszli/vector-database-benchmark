import unittest
import numpy as np
import paddle
from paddle.base import core

class SimpleNet(paddle.nn.Layer):

    def __init__(self, input_size, output_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, output_size)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.linear(x)
        return x

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_float16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the float16')
@unittest.skipIf(not core.is_compiled_with_cuda() or paddle.device.cuda.get_device_capability()[0] < 7.0, "run test when gpu's compute capability is at least 7.0.")
class TestMasterGrad(unittest.TestCase):

    def check_results(self, fp32_grads, op_list, total_steps, accumulate_batchs_num):
        if False:
            print('Hello World!')
        for grad in fp32_grads:
            self.assertEqual(grad.dtype, paddle.float32)
        self.assertEqual(int(op_list['matmul_v2'].split(',')[0]), total_steps)
        self.assertEqual(int(op_list['adam_'].split(',')[0]), 2 * (total_steps / accumulate_batchs_num))
        self.assertEqual(int(op_list['transfer_dtype'].split(',')[0]), total_steps + total_steps * 2 + 2)

    def run_dygraph(self, total_steps, accumulate_batchs_num, model, optimizer):
        if False:
            print('Hello World!')
        (model, opt) = paddle.amp.decorate(model, optimizers=optimizer, level='O2', master_grad=True)
        scaler = paddle.amp.GradScaler()
        paddle.amp.debugging.enable_operator_stats_collection()
        for i in range(total_steps):
            x = np.random.random((2, 2)).astype('float32')
            label = np.random.random((2, 4)).astype('float32')
            with paddle.amp.auto_cast(level='O2'):
                out = model(paddle.to_tensor(x))
                loss = paddle.nn.functional.l1_loss(out, paddle.to_tensor(label))
            scaled = scaler.scale(loss)
            scaled.backward()
            fp32_grads = [model.linear.weight.grad, model.linear.bias.grad]
            if (i + 1) % accumulate_batchs_num == 0:
                scaler.step(opt)
                scaler.update()
                opt.clear_grad()
        paddle.amp.debugging.disable_operator_stats_collection()
        op_list = paddle.base.core.get_low_precision_op_list()
        return (fp32_grads, op_list)

    def test_adam_master_grad(self):
        if False:
            i = 10
            return i + 15
        total_steps = 4
        accumulate_batchs_num = 2
        model = SimpleNet(2, 4)
        opt = paddle.optimizer.Adam(parameters=model.parameters())
        (fp32_grads, op_list) = self.run_dygraph(total_steps, accumulate_batchs_num, model, opt)
        self.check_results(fp32_grads, op_list, total_steps, accumulate_batchs_num)

    def test_momentum_master_grad(self):
        if False:
            i = 10
            return i + 15
        total_steps = 4
        accumulate_batchs_num = 1
        model = SimpleNet(2, 4)
        L1Decay = paddle.regularizer.L1Decay(0.0001)
        opt = paddle.optimizer.Momentum(parameters=model.parameters(), weight_decay=L1Decay)
        (fp32_grads, op_list) = self.run_dygraph(total_steps, accumulate_batchs_num, model, opt)
        for grad in fp32_grads:
            self.assertEqual(grad.dtype, paddle.float32)
if __name__ == '__main__':
    unittest.main()
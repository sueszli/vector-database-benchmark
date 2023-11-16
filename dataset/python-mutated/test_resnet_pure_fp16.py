import time
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
from test_resnet import SEED, ResNet, optimizer_setting
import paddle
from paddle import base
from paddle.base import core
batch_size = 2
epoch_num = 1
if base.is_compiled_with_cuda():
    base.set_flags({'FLAGS_cudnn_deterministic': True})

def train(to_static, build_strategy=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests model decorated by `dygraph_to_static_output` in static graph mode. For users, the model is defined in dygraph mode and trained in static graph mode.\n    '
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    resnet = ResNet()
    if to_static:
        resnet = paddle.jit.to_static(resnet, build_strategy=build_strategy)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    (resnet, optimizer) = paddle.amp.decorate(models=resnet, optimizers=optimizer, level='O2', save_dtype='float32')
    for epoch in range(epoch_num):
        loss_data = []
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        for batch_id in range(100):
            start_time = time.time()
            img = paddle.to_tensor(np.random.random([batch_size, 3, 224, 224]).astype('float32'))
            label = paddle.to_tensor(np.random.randint(0, 100, [batch_size, 1], dtype='int64'))
            img.stop_gradient = True
            label.stop_gradient = True
            with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
                pred = resnet(img)
                loss = paddle.nn.functional.cross_entropy(input=pred, label=label, reduction='none', use_softmax=False)
            avg_loss = paddle.mean(x=pred)
            acc_top1 = paddle.static.accuracy(input=pred, label=label, k=1)
            acc_top5 = paddle.static.accuracy(input=pred, label=label, k=5)
            scaled = scaler.scale(avg_loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            resnet.clear_gradients()
            loss_data.append(float(avg_loss))
            total_loss += avg_loss
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1
            end_time = time.time()
            if batch_id % 2 == 0:
                print('epoch %d | batch step %d, loss %0.3f, acc1 %0.3f, acc5 %0.3f, time %f' % (epoch, batch_id, total_loss.numpy() / total_sample, total_acc1.numpy() / total_sample, total_acc5.numpy() / total_sample, end_time - start_time))
            if batch_id == 10:
                break
    return loss_data

class TestResnet(Dy2StTestBase):

    def train(self, to_static):
        if False:
            for i in range(10):
                print('nop')
        paddle.jit.enable_to_static(to_static)
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.enable_inplace = False
        return train(to_static, build_strategy)

    @test_legacy_and_pir
    def test_resnet(self):
        if False:
            while True:
                i = 10
        if base.is_compiled_with_cuda():
            static_loss = self.train(to_static=True)
            dygraph_loss = self.train(to_static=False)
            np.testing.assert_allclose(static_loss, dygraph_loss, rtol=1e-05, atol=0.001, err_msg=f'static_loss: {static_loss} \n dygraph_loss: {dygraph_loss}')

    def test_resnet_composite(self):
        if False:
            return 10
        if base.is_compiled_with_cuda():
            core._set_prim_backward_enabled(True)
            static_loss = self.train(to_static=True)
            core._set_prim_backward_enabled(False)
            dygraph_loss = self.train(to_static=False)
            np.testing.assert_allclose(static_loss, dygraph_loss, rtol=1e-05, atol=0.001, err_msg=f'static_loss: {static_loss} \n dygraph_loss: {dygraph_loss}')
if __name__ == '__main__':
    unittest.main()
import time
import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.vision.models import resnet50
SEED = 2020
base_lr = 0.001
momentum_rate = 0.9
l2_decay = 0.0001
batch_size = 2
epoch_num = 1
DY2ST_PRIM_GT = [5.847333908081055, 8.368712425231934, 4.989010334014893, 8.523179054260254, 7.997398376464844, 7.601831436157227, 9.777579307556152, 8.428393363952637, 8.581992149353027, 10.313587188720703]
if core.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})

def reader_decorator(reader):
    if False:
        for i in range(10):
            print('nop')

    def __reader__():
        if False:
            for i in range(10):
                print('nop')
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield (img, label)
    return __reader__

class TransedFlowerDataSet(paddle.io.Dataset):

    def __init__(self, flower_data, length):
        if False:
            for i in range(10):
                print('nop')
        self.img = []
        self.label = []
        self.flower_data = flower_data()
        self._generate(length)

    def _generate(self, length):
        if False:
            for i in range(10):
                print('nop')
        for (i, data) in enumerate(self.flower_data):
            if i >= length:
                break
            self.img.append(data[0])
            self.label.append(data[1])

    def __getitem__(self, idx):
        if False:
            return 10
        return (self.img[idx], self.label[idx])

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.img)

def optimizer_setting(parameter_list=None):
    if False:
        print('Hello World!')
    optimizer = paddle.optimizer.Momentum(learning_rate=base_lr, momentum=momentum_rate, weight_decay=paddle.regularizer.L2Decay(l2_decay), parameters=parameter_list)
    return optimizer

def run(model, data_loader, optimizer, mode):
    if False:
        while True:
            i = 10
    if mode == 'train':
        model.train()
        end_step = 9
    elif mode == 'eval':
        model.eval()
        end_step = 1
    for epoch in range(epoch_num):
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        losses = []
        for (batch_id, data) in enumerate(data_loader()):
            start_time = time.time()
            (img, label) = data
            pred = model(img)
            avg_loss = paddle.nn.functional.cross_entropy(input=pred, label=label, soft_label=False, reduction='mean', use_softmax=True)
            acc_top1 = paddle.static.accuracy(input=pred, label=label, k=1)
            acc_top5 = paddle.static.accuracy(input=pred, label=label, k=5)
            if mode == 'train':
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1
            losses.append(avg_loss.numpy().item())
            end_time = time.time()
            print('[%s]epoch %d | batch step %d, loss %0.8f, acc1 %0.3f, acc5 %0.3f, time %f' % (mode, epoch, batch_id, avg_loss, total_acc1.numpy() / total_sample, total_acc5.numpy() / total_sample, end_time - start_time))
            if batch_id >= end_step:
                break
    print(losses)
    return losses

def train(to_static, enable_prim, enable_cinn):
    if False:
        i = 10
        return i + 15
    if core.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    base.core._set_prim_all_enabled(enable_prim)
    dataset = TransedFlowerDataSet(reader_decorator(paddle.dataset.flowers.train(use_xmap=False)), batch_size * (10 + 1))
    data_loader = paddle.io.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    resnet = resnet50(False)
    if to_static:
        build_strategy = paddle.static.BuildStrategy()
        if enable_cinn:
            build_strategy.build_cinn_pass = True
        resnet = paddle.jit.to_static(resnet, build_strategy=build_strategy, full_graph=True)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    train_losses = run(resnet, data_loader, optimizer, 'train')
    if to_static and enable_prim and enable_cinn:
        eval_losses = run(resnet, data_loader, optimizer, 'eval')
    return train_losses

class TestResnet(unittest.TestCase):

    @unittest.skipIf(not (paddle.is_compiled_with_cinn() and paddle.is_compiled_with_cuda()), 'paddle is not compiled with CINN and CUDA')
    def test_prim(self):
        if False:
            while True:
                i = 10
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)
        np.testing.assert_allclose(dy2st_prim, DY2ST_PRIM_GT, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()
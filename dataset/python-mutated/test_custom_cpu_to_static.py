import os
import sys
import tempfile
import time
import unittest
import numpy as np
EPOCH_NUM = 1
BATCH_SIZE = 1024

def train_func_base(epoch_id, train_loader, model, cost, optimizer):
    if False:
        for i in range(10):
            print('nop')
    total_step = len(train_loader)
    epoch_start = time.time()
    for (batch_id, (images, labels)) in enumerate(train_loader()):
        outputs = model(images)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(epoch_id + 1, EPOCH_NUM, batch_id + 1, total_step, loss.numpy()))
    epoch_end = time.time()
    print(f'Epoch ID: {epoch_id + 1}, FP32 train epoch time: {(epoch_end - epoch_start) * 1000} ms')

def train_func_ampo1(epoch_id, train_loader, model, cost, optimizer, scaler):
    if False:
        return 10
    import paddle
    total_step = len(train_loader)
    epoch_start = time.time()
    for (batch_id, (images, labels)) in enumerate(train_loader()):
        with paddle.amp.auto_cast(custom_black_list={'flatten_contiguous_range', 'greater_than', 'matmul_v2'}, level='O1'):
            outputs = model(images)
            loss = cost(outputs, labels)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(epoch_id + 1, EPOCH_NUM, batch_id + 1, total_step, loss.numpy()))
    epoch_end = time.time()
    print(f'Epoch ID: {epoch_id + 1}, AMPO1 train epoch time: {(epoch_end - epoch_start) * 1000} ms')

def test_func(epoch_id, test_loader, model, cost):
    if False:
        i = 10
        return i + 15
    import paddle
    model.eval()
    avg_acc = [[], []]
    for (batch_id, (images, labels)) in enumerate(test_loader()):
        outputs = model(images)
        loss = cost(outputs, labels)
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        avg_acc[0].append(acc_top1.numpy())
        avg_acc[1].append(acc_top5.numpy())
    model.train()
    print(f'Epoch ID: {epoch_id + 1}, Top1 accurary: {np.array(avg_acc[0]).mean()}, Top5 accurary: {np.array(avg_acc[1]).mean()}')

class TestCustomCPUPlugin(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {}             && git clone --depth 1 {}             && cd PaddleCustomDevice             && git fetch origin             && git checkout {} -b dev             && cd backends/custom_cpu             && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} -DWITH_TESTING=OFF && make -j8'.format(self.temp_dir.name, os.getenv('PLUGIN_URL'), os.getenv('PLUGIN_TAG'), sys.executable)
        os.system(cmd)
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(cur_dir, f'{self.temp_dir.name}/PaddleCustomDevice/backends/custom_cpu/build')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_custom_cpu_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_to_static()
        self._test_amp_o1()

    def _test_to_static(self):
        if False:
            i = 10
            return i + 15
        import paddle

        class LeNet5(paddle.nn.Layer):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc = paddle.nn.Linear(in_features=1024, out_features=10)
                self.relu = paddle.nn.ReLU()
                self.fc1 = paddle.nn.Linear(in_features=10, out_features=10)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = paddle.flatten(x, 1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                return out
        paddle.set_device('custom_cpu')
        model = LeNet5()
        cost = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        build_strategy = paddle.static.BuildStrategy()
        mnist = paddle.jit.to_static(model, build_strategy=build_strategy)
        transform = paddle.vision.transforms.Compose([paddle.vision.transforms.Resize((32, 32)), paddle.vision.transforms.ToTensor(), paddle.vision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True)
        test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform, download=True)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
        for epoch_id in range(EPOCH_NUM):
            train_func_base(epoch_id, train_loader, model, cost, optimizer)
            test_func(epoch_id, test_loader, model, cost)

    def _test_amp_o1(self):
        if False:
            while True:
                i = 10
        import paddle

        class LeNet5(paddle.nn.Layer):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fc = paddle.nn.Linear(in_features=1024, out_features=10)
                self.relu = paddle.nn.ReLU()
                self.fc1 = paddle.nn.Linear(in_features=10, out_features=10)

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = paddle.flatten(x, 1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                return out
        paddle.set_device('custom_cpu')
        model = LeNet5()
        cost = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        (model, optimizer) = paddle.amp.decorate(models=model, optimizers=optimizer, level='O1')
        transform = paddle.vision.transforms.Compose([paddle.vision.transforms.Resize((32, 32)), paddle.vision.transforms.ToTensor(), paddle.vision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True)
        test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform, download=True)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
        for epoch_id in range(EPOCH_NUM):
            train_func_ampo1(epoch_id, train_loader, model, cost, optimizer, scaler)
            test_func(epoch_id, test_loader, model, cost)
if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        sys.exit()
    unittest.main()
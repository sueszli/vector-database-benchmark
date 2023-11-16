import numpy as np
import paddle
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
from paddle.nn import Linear, ReLU
seed = 2022
epoch = 2
linear_size = 1000
np.random.seed(seed)
paddle.seed(seed)

class MLP(paddle.nn.Layer):

    def __init__(self, linear_size=1000):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)
        self._relu = ReLU()

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        y = self._relu(y)
        return y

class RandomDataset(paddle.io.Dataset):

    def __init__(self, num_samples=200, linear_size=1000):
        if False:
            print('Hello World!')
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        img = np.random.rand(self.linear_size).astype('float32')
        return img

    def __len__(self):
        if False:
            print('Hello World!')
        return self.num_samples

def optimizer_setting(model, use_pure_bf16, use_main_grad):
    if False:
        while True:
            i = 10
    if use_main_grad:
        assert use_pure_bf16
        model = mix_precision_utils.MixPrecisionLayer(model, dtype='bfloat16')
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=1e-05, weight_decay=1e-05, grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0), multi_precision=use_pure_bf16)
    if use_main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)
    return optimizer

def train_mlp(model, use_pure_bf16=False, use_main_grad=False, accumulate_grad=False):
    if False:
        while True:
            i = 10
    optimizer = optimizer_setting(model=model, use_pure_bf16=use_pure_bf16, use_main_grad=use_main_grad)
    if use_pure_bf16:
        level = 'O2'
        custom_white_list = None
        model = paddle.amp.decorate(models=model, dtype='bfloat16', level=level)
    else:
        level = 'O1'
        custom_white_list = ['matmul_v2', 'elementwise_add', 'relu', 'reduce_mean']
    model = paddle.DataParallel(model)
    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(RandomDataset(), batch_size=100, shuffle=False, drop_last=True, num_workers=0)
    if not use_pure_bf16:
        for param in model.parameters():
            t = paddle.cast(paddle.cast(param, dtype='bfloat16'), dtype='float32')
            param.set_value(t)
    losses = []
    for eop in range(epoch):
        model.train()
        for (batch_id, data) in enumerate(train_loader()):
            data.stop_gradient = True
            with model.no_sync():
                with paddle.amp.auto_cast(True, level=level, dtype='bfloat16', custom_white_list=custom_white_list):
                    out = model(data)
                    loss = paddle.mean(out)
                losses.append(loss)
                loss.backward()
            if not accumulate_grad:
                fused_allreduce_gradients(list(model.parameters()), None)
                optimizer.step()
                optimizer.clear_grad()
        if accumulate_grad:
            fused_allreduce_gradients(list(model.parameters()), None)
            optimizer.step()
            optimizer.clear_grad()
    return losses

def test_dp_bf16():
    if False:
        i = 10
        return i + 15
    if not paddle.amp.is_bfloat16_supported():
        return
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = MLP()
    mlp2 = MLP()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)
    losses_o1 = train_mlp(mlp1, use_pure_bf16=False)
    losses_o2 = train_mlp(mlp2, use_pure_bf16=True, use_main_grad=True)
    for i in range(len(losses_o2)):
        loss_o2 = paddle.cast(losses_o2[i], dtype='float32').detach()
        loss_o1 = paddle.cast(losses_o1[i], dtype='float32').detach()
        np.testing.assert_array_equal(loss_o2, loss_o1)
    mlp3 = MLP()
    mlp4 = MLP()
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)
    losses_acc_grad_o1 = train_mlp(mlp3, use_pure_bf16=False, accumulate_grad=True)
    losses_acc_grad_o2 = train_mlp(mlp4, use_pure_bf16=True, use_main_grad=True, accumulate_grad=True)
    for i in range(len(losses_acc_grad_o2)):
        loss_acc_grad_o2 = paddle.cast(losses_acc_grad_o2[i], dtype='float32').detach()
        loss_acc_grad_o1 = paddle.cast(losses_acc_grad_o1[i], dtype='float32').detach()
        np.testing.assert_array_equal(loss_acc_grad_o2, loss_acc_grad_o1)
if __name__ == '__main__':
    test_dp_bf16()
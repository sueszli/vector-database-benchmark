import numpy as np
from test_collective_multi_nodes import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import nn
from paddle.distributed import fleet

def weight_init(mp, shape, col=True, seed=1024):
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(seed)
    w = np.random.normal(0, 0.02, size=shape)
    if mp is None:
        _w = w
    elif col:
        step = shape[1] // mp.nranks
        _w = w[:, mp.rank * step:mp.rank * step + step]
    else:
        step = shape[0] // mp.nranks
        _w = w[mp.rank * step:mp.rank * step + step, :]
    return paddle.nn.initializer.Assign(_w)

class Criterion(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.loss_func = nn.MSELoss(reduction='mean')

    def forward(self, pred, label):
        if False:
            i = 10
            return i + 15
        loss = self.loss_func(pred, label)
        return loss

class ModelPipeline(fleet.meta_parallel.PipelineLayer):

    def __init__(self, hcg):
        if False:
            while True:
                i = 10
        paddle.seed(1024)
        dp_linear = nn.Linear(32, 128)
        self.layers_pp = []
        self.topology = hcg.topology()
        self.layers_pp.append(dp_linear)
        mp = hcg.get_model_parallel_group()
        for i in range(6):
            if mp is not None and mp.nranks > 1:
                mp_linear_1 = fleet.meta_parallel.ColumnParallelLinear(128, 512, weight_attr=weight_init(mp, (128, 512), True, 1204 + i), has_bias=True, gather_output=False)
                mp_linear_2 = fleet.meta_parallel.RowParallelLinear(512, 128, weight_attr=weight_init(mp, (512, 128), False, 2012 + i), has_bias=True, input_is_parallel=True)
            else:
                mp_linear_1 = nn.Linear(128, 512, weight_attr=weight_init(None, (128, 512), True, 1204 + i))
                mp_linear_2 = nn.Linear(512, 128, weight_attr=weight_init(None, (512, 128), True, 2012 + i))
            act = nn.ReLU6()
            layer_seq = nn.Sequential(mp_linear_1, mp_linear_2, act)
            self.layers_pp.append(layer_seq)
        out = nn.Linear(128, 32)
        self.layers_pp.append(out)
        super().__init__(layers=self.layers_pp, loss_fn=Criterion(), topology=self.topology)

class Model(nn.Layer):

    def __init__(self, hcg):
        if False:
            print('Hello World!')
        super().__init__()
        paddle.seed(1024)
        dp_linear = nn.Linear(32, 128)
        self.layers_pp = []
        self.layers_pp.append(dp_linear)
        mp = hcg.get_model_parallel_group() if hcg else None
        for i in range(6):
            if mp is not None and mp.nranks > 1:
                mp_linear_1 = fleet.meta_parallel.ColumnParallelLinear(128, 512, weight_attr=weight_init(mp, (128, 512), True, 1204 + i), has_bias=True, gather_output=False)
                mp_linear_2 = fleet.meta_parallel.RowParallelLinear(512, 128, weight_attr=weight_init(mp, (512, 128), False, 2012 + i), has_bias=True, input_is_parallel=True)
            else:
                mp_linear_1 = nn.Linear(128, 512, weight_attr=weight_init(None, (128, 512), True, 1204 + i))
                mp_linear_2 = nn.Linear(512, 128, weight_attr=weight_init(None, (512, 128), True, 2012 + i))
            act = nn.ReLU6()
            layer_seq = nn.Sequential(mp_linear_1, mp_linear_2, act)
            self.layers_pp.append(layer_seq)
        out = nn.Linear(128, 32)
        self.layers_pp.append(out)
        self.layers = nn.Sequential(*self.layers_pp)

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.layers(x)

class TestDygrapgHybridDPPPMP(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def check_pass(self, *args, **kwargs):
        if False:
            return 10
        from common import init_parallel_env
        import paddle
        from paddle.distributed import fleet
        hcg = init_parallel_env('DP4-MP2-PP2-SH1-O1', 64)
        pp_degree = hcg.get_pipe_parallel_world_size()
        import numpy as np
        crit = Criterion()
        if pp_degree <= 1:
            model = Model(hcg)
        else:
            model = ModelPipeline(hcg)
        model_base = Model(None)
        optimizer = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
        optimizer_base = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_base.parameters())
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        loss_hybrid_arr = []
        loss_base_arr = []
        x = paddle.to_tensor(np.random.random((16, 32))).astype('float32')
        y = paddle.to_tensor(np.random.random((16, 32))).astype('float32')
        for _ in range(5):
            if pp_degree > 1:
                loss = model.train_batch([x, y], optimizer=optimizer)
            else:
                output = model(x)
                loss = crit(output, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            output_base = model_base(x)
            loss_base = crit(output_base, y)
            loss_base.backward()
            optimizer_base.step()
            optimizer_base.clear_grad()
            loss_base_arr.append(loss_base.numpy())
            loss_hybrid_arr.append(loss.numpy())
        np.testing.assert_allclose(loss_base_arr, loss_hybrid_arr, rtol=1e-05, atol=1e-05)
if __name__ == '__main__':
    runtime_main(TestDygrapgHybridDPPPMP, 'dpppmp')
import os
from auto_parallel.semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist

class TestSimpleNetHybridStrategyForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        paddle.set_device(self._backend)
        self.set_random_seed(self._seed)
        self.init_single_card_net_result()

    def test_dp_mp_demo_net(self):
        if False:
            return 10
        self.set_random_seed(self._seed)
        model = dist.shard_layer(DemoNet('dp_mp_hybrid_strategy'), self._mesh, self.shard_fn)
        (self.dp_mp_loss, self.dp_mp_parameters) = self.run_dynamic(model, shard_input=True)
        self.check_tensor_eq(self.dp_mp_loss, self.base_loss)
        for (param, param_base) in zip(self.dp_mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_dp_mp_demo_net()
if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
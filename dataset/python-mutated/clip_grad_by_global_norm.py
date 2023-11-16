import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto
paddle.enable_static()

def apply_pass(use_sharding=False):
    if False:
        i = 10
        return i + 15
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_sharding:
        sharding = strategy.sharding
        sharding.enable = True
        sharding.degree = 2
        sharding.stage = 2
    return strategy

def get_parameter_value(program):
    if False:
        for i in range(10):
            print('nop')
    from paddle.base.framework import Parameter

    def is_parameter(var):
        if False:
            while True:
                i = 10
        return isinstance(var, Parameter)

    def get_tensor(var):
        if False:
            print('Hello World!')
        t = paddle.base.global_scope().find_var(var.name).get_tensor()
        return np.array(t)

    def get_name(var):
        if False:
            for i in range(10):
                print('nop')
        return len(var.name)
    parameters_list = list(filter(is_parameter, program.list_vars()))
    parameters_value = []
    for p in sorted(parameters_list, key=get_name):
        parameters_value.append(get_tensor(p))
    return parameters_value

def reset_prog():
    if False:
        print('Hello World!')
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestGradientClipByGlobalNorm(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.batch_size = 2
        self.batch_num = 1
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            print('Hello World!')
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_sharding=False):
        if False:
            print('Hello World!')
        reset_prog()
        strategy = apply_pass(use_sharding)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('dp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_result(self, dp_params, sharding_params):
        if False:
            i = 10
            return i + 15
        assert len(dp_params) == len(sharding_params)
        for (dp_p, sharding_p) in zip(dp_params, sharding_params):
            np.testing.assert_allclose(dp_p, sharding_p, rtol=1e-05, atol=1e-08, err_msg='gradient clip by global norm has wrong results!, \nu={}\nv={}\ndiff={}'.format(dp_p, sharding_p, dp_p - sharding_p))

    def test_grad_clip(self):
        if False:
            return 10
        dp_engine = self.get_engine()
        dp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        dp_param_values = get_parameter_value(dp_engine.main_program)
        sharding_engine = self.get_engine(True)
        sharding_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        sharding_param_values = get_parameter_value(sharding_engine.main_program)
        self.check_result(dp_param_values, sharding_param_values)
if __name__ == '__main__':
    unittest.main()
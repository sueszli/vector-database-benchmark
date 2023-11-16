import unittest
import numpy as np
import paddle
from paddle import static, utils

def gen_data():
    if False:
        i = 10
        return i + 15
    return np.random.random(size=(10, 5)).astype('float32')

class TestFleetStaticEMA(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))
        self._ema_decay = 0.999
        self._param_name = 'fc.weight'
        self._train_program = static.Program()
        self._startup_prog = static.Program()
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        paddle.distributed.fleet.init(is_collective=True, strategy=strategy)
        with static.program_guard(self._train_program, self._startup_prog):
            with utils.unique_name.guard():
                data = static.data(name='x', shape=[-1, 5], dtype='float32')
                hidden = static.nn.fc(x=data, size=10, weight_attr=self._param_name)
                cost = paddle.mean(hidden)
                self._test_program = static.default_main_program().clone(for_test=True)
                optimizer = paddle.optimizer.Adam(learning_rate=0.001)
                optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer, strategy)
                optimizer.minimize(cost)
                self._ema = static.ExponentialMovingAverage(self._ema_decay)
                self._ema.update()

    def train(self, place, restore):
        if False:
            while True:
                i = 10
        exe = static.Executor(place)
        exe.run(self._startup_prog)
        params = []
        for pass_id in range(2):
            for batch_id in range(3):
                exe.run(program=self._train_program, feed={'x': gen_data()})
                tmp_param = np.array(static.global_scope().find_var(self._param_name).get_tensor())
                params.append(tmp_param)
            with self._ema.apply(exe, restore):
                final_ema = np.array(static.global_scope().find_var(self._param_name).get_tensor())
                exe.run(program=self._test_program, feed={'x': gen_data()})
            if not restore:
                self._ema.restore(exe)
        return (params, final_ema)

    def test_check_ema(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self._places:
            for restore in (True, False):
                (params, final_ema) = self.train(place, restore)
                manu_ema = np.zeros_like(final_ema)
                if len(params) > 0:
                    for param in params:
                        manu_ema = self._ema_decay * manu_ema + (1 - self._ema_decay) * param
                    manu_ema = manu_ema / (1.0 - self._ema_decay ** len(params))
                np.testing.assert_allclose(manu_ema, final_ema, rtol=1e-05)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
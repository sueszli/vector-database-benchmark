import unittest
import paddle
from paddle.incubate.layers.nn import pow2_decay_with_linear_warmup
from paddle.optimizer.lr import LinearWarmup, PolynomialDecay

def gen_pow2_warmup_op_lr(warmup_steps, total_steps, base_lr, end_lr, place):
    if False:
        print('Hello World!')
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        lr = pow2_decay_with_linear_warmup(warmup_steps, total_steps, base_lr, end_lr)
        exe = paddle.static.Executor(place)
    with paddle.static.scope_guard(paddle.static.Scope()):
        exe.run(startup)
        while True:
            lr_np = exe.run(main, fetch_list=[lr])[0]
            yield lr_np[0]

class Pow2Warmup(LinearWarmup):

    def __init__(self, warmup_steps, total_steps, base_lr, end_lr):
        if False:
            print('Hello World!')
        assert total_steps > warmup_steps
        lr_sch = PolynomialDecay(learning_rate=base_lr, decay_steps=total_steps - warmup_steps, end_lr=end_lr, power=2)
        super().__init__(learning_rate=lr_sch, warmup_steps=warmup_steps, start_lr=0.0, end_lr=base_lr)

def gen_pow2_warmup_py_lr(warmup_steps, total_steps, base_lr, end_lr, place):
    if False:
        i = 10
        return i + 15
    lr_sch = Pow2Warmup(warmup_steps, total_steps, base_lr, end_lr)
    lr_sch.step()
    while True:
        yield lr_sch()
        lr_sch.step()

class TestPow2WarmupLRScheduler(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.params = {'warmup_steps': 30, 'total_steps': 100, 'base_lr': 0.02, 'end_lr': 0.001}
        self.step_num = 1000

    def check_with_place(self, place):
        if False:
            i = 10
            return i + 15
        kwargs = dict(self.params)
        kwargs['place'] = place
        lr_sch_op = gen_pow2_warmup_op_lr(**kwargs)
        lr_sch_py = gen_pow2_warmup_py_lr(**kwargs)
        for (i, (lr_op, lr_py)) in enumerate(zip(lr_sch_op, lr_sch_py)):
            self.assertLess(abs(lr_op - lr_py), 1e-06)
            if i > self.step_num:
                break

    def test_main(self):
        if False:
            print('Hello World!')
        self.check_with_place(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.check_with_place(paddle.CUDAPlace(0))
if __name__ == '__main__':
    unittest.main()
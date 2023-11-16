import os
import unittest
import numpy
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from simple_nets import fc_with_batchnorm, init_data, simple_fc_net
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

class TestMNIST(TestParallelExecutorBase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        os.environ['CPU_NUM'] = str(4)

    def _compare_fuse_elewise_add_act_ops(self, model, use_device):
        if False:
            print('Hello World!')
        if use_device == DeviceType.CUDA and (not core.is_compiled_with_cuda()):
            return
        (img, label) = init_data()

        def _optimizer(learning_rate=1e-06):
            if False:
                for i in range(10):
                    print('nop')
            optimizer = paddle.optimizer.SGD(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(1e-06))
            return optimizer
        (not_fuse_op_first_loss, not_fuse_op_last_loss, _) = self.check_network_convergence(model, feed_dict={'image': img, 'label': label}, use_device=use_device, fuse_elewise_add_act_ops=False, use_ir_memory_optimize=False, enable_inplace=False, optimizer=_optimizer)
        (fuse_op_first_loss, fuse_op_last_loss, _) = self.check_network_convergence(model, feed_dict={'image': img, 'label': label}, use_device=use_device, fuse_elewise_add_act_ops=True, use_ir_memory_optimize=False, enable_inplace=False, optimizer=_optimizer)
        self.assertAlmostEqual(not_fuse_op_first_loss, fuse_op_first_loss, delta=1e-06)
        self.assertAlmostEqual(not_fuse_op_last_loss, fuse_op_last_loss, delta=1e-06)

    def test_simple_fc_with_fuse_op(self):
        if False:
            return 10
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, DeviceType.CUDA)
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, DeviceType.CPU)

    def test_batchnorm_fc_with_fuse_op(self):
        if False:
            for i in range(10):
                print('nop')
        self._compare_fuse_elewise_add_act_ops(fc_with_batchnorm, DeviceType.CUDA)
        self._compare_fuse_elewise_add_act_ops(fc_with_batchnorm, DeviceType.CPU)

class TestFuseActElewiseAddInplaceGradPass(unittest.TestCase):

    def build_program(self, main_program, startup_program):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(main_program, startup_program):
            X = paddle.static.data(name='X', shape=[3, 3], dtype='float32')
            Y = paddle.static.data(name='Y', shape=[3, 3], dtype='float32')
            Out1 = X * 5
            Out2 = F.relu(Out1)
            prediction = paddle.tensor.math._add_with_axis(Y, Out2, axis=1)
            loss = paddle.mean(prediction)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)
        return (X, Y, loss)

    def check(self, place):
        if False:
            print('Hello World!')
        paddle.seed(1)
        numpy.random.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_program = base.Program()
        startup_program = base.Program()
        (X, Y, loss) = self.build_program(main_program, startup_program)
        exe = base.Executor(place)
        x = numpy.random.random(size=(3, 3)).astype('float32')
        y = numpy.random.random(size=(3, 3)).astype('float32')
        label = numpy.random.random(size=(3, 3)).astype('float32')
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = True
        compiled_prog_fused = paddle.static.CompiledProgram(main_program, build_strategy=build_strategy)
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            loss_data_fused = exe.run(compiled_prog_fused, feed={'X': x, 'Y': y}, fetch_list=[loss.name])
        build_strategy = base.BuildStrategy()
        build_strategy.fuse_elewise_add_act_ops = False
        compiled_prog = paddle.static.CompiledProgram(main_program, build_strategy=build_strategy)
        scope = base.Scope()
        with base.scope_guard(scope):
            exe.run(startup_program)
            loss_data = exe.run(compiled_prog, feed={'X': x, 'Y': y}, fetch_list=[loss.name])
        self.assertEqual(loss_data_fused, loss_data)

    def test_fuse_act_add_grad_pass_cpu(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        self.check(place)

    def test_fuse_act_add_grad_pass_cuda(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self.check(place)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
import unittest
import numpy as np
from simple_nets import simple_fc_net_with_inputs
import paddle
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.device.cuda.graphs import CUDAGraph

def can_use_cuda_graph():
    if False:
        for i in range(10):
            print('nop')
    return paddle.is_compiled_with_cuda() and (not paddle.is_compiled_with_rocm())

def build_program(main, startup, batch_size, class_num):
    if False:
        print('Hello World!')
    image_shape = [batch_size, 784]
    label_shape = [batch_size, 1]
    with paddle.static.program_guard(main, startup):
        image = paddle.static.data(name='image', shape=image_shape, dtype='float32')
        label = paddle.static.data(name='label', shape=label_shape, dtype='int64')
        image.persistable = True
        label.persistable = True
        loss = simple_fc_net_with_inputs(image, label, class_num)
        loss.persistable = True
        lr = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04])
        optimizer = paddle.optimizer.SGD(learning_rate=lr)
        optimizer.minimize(loss)
    return (image, label, loss, lr)

@unittest.skipIf(not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0, 'only support cuda >= 11.0')
class TestCUDAGraphInStaticMode(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        if can_use_cuda_graph():
            paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth', 'FLAGS_sync_nccl_allreduce': False, 'FLAGS_cudnn_deterministic': True, 'FLAGS_use_stream_safe_cuda_allocator': True})

    @switch_to_static_graph
    def test_cuda_graph_static_graph(self):
        if False:
            i = 10
            return i + 15
        if not can_use_cuda_graph():
            return
        seed = 100
        loss_cuda_graph = self.cuda_graph_static_graph_main(seed, use_cuda_graph=True)
        loss_no_cuda_graph = self.cuda_graph_static_graph_main(seed, use_cuda_graph=False)
        self.assertEqual(loss_cuda_graph, loss_no_cuda_graph)

    def cuda_graph_static_graph_main(self, seed, use_cuda_graph):
        if False:
            i = 10
            return i + 15
        batch_size = 1
        class_num = 10
        image_shape = [batch_size, 784]
        label_shape = [batch_size, 1]
        paddle.seed(seed)
        np.random.seed(seed)
        startup = paddle.static.Program()
        main = paddle.static.Program()
        (image, label, loss, lr) = build_program(main, startup, batch_size, class_num)
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.allow_cuda_graph_capture = True
            build_strategy.fix_op_run_order = True
            build_strategy.fuse_all_optimizer_ops = True
            compiled_program = paddle.static.CompiledProgram(main, build_strategy=build_strategy)
            image_t = scope.var(image.name).get_tensor()
            label_t = scope.var(label.name).get_tensor()
            loss_t = scope.var(loss.name).get_tensor()
            lr_var = main.global_block().var(lr._var_name)
            self.assertTrue(lr_var.persistable)
            lr_t = scope.var(lr_var.name).get_tensor()
            cuda_graph = None
            for batch_id in range(20):
                image_np = np.random.rand(*image_shape).astype('float32')
                label_np = np.random.randint(low=0, high=class_num, size=label_shape, dtype='int64')
                image_t.set(image_np, place)
                label_t.set(label_np, place)
                if batch_id == 1 and use_cuda_graph:
                    cuda_graph = CUDAGraph(place, mode='global')
                    cuda_graph.capture_begin()
                    exe.run(compiled_program)
                    cuda_graph.capture_end()
                if cuda_graph:
                    lr_t.set(np.array([lr()], dtype='float32'), place)
                    cuda_graph.replay()
                else:
                    exe.run(compiled_program)
                lr.step()
            if cuda_graph:
                cuda_graph.reset()
        return np.array(loss_t)
if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from test_cuda_graph_static_mode import build_program, can_use_cuda_graph
import paddle
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.device.cuda.graphs import CUDAGraph

@unittest.skipIf(not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0, 'only support cuda >= 11.0')
class TestCUDAGraphInFirstBatch(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        if can_use_cuda_graph():
            paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth', 'FLAGS_sync_nccl_allreduce': False, 'FLAGS_cudnn_deterministic': True, 'FLAGS_use_stream_safe_cuda_allocator': True})

    @switch_to_static_graph
    def test_cuda_graph_in_first_batch(self):
        if False:
            for i in range(10):
                print('nop')
        if not can_use_cuda_graph():
            return
        startup = paddle.static.Program()
        main = paddle.static.Program()
        (image, label, loss, lr) = build_program(main, startup, 1, 10)
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.allow_cuda_graph_capture = True
            compiled_program = paddle.static.CompiledProgram(main, build_strategy=build_strategy)
            cuda_graph = None
            image_t = scope.var(image.name).get_tensor()
            label_t = scope.var(label.name).get_tensor()
            image_np = np.random.rand(1, 784).astype('float32')
            label_np = np.random.randint(low=0, high=10, size=[1, 1], dtype='int64')
            image_t.set(image_np, place)
            label_t.set(label_np, place)
            with self.assertRaises(RuntimeError):
                cuda_graph = CUDAGraph(place, mode='global')
                cuda_graph.capture_begin()
                exe.run(compiled_program)
                cuda_graph.capture_end()
                if cuda_graph:
                    cuda_graph.reset()
if __name__ == '__main__':
    unittest.main()
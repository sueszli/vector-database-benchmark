import unittest
from op_test import OpTestTool
import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import IrGraph, Program, program_guard
from paddle.static.quantization import QuantizationTransformPass
paddle.enable_static()

class TestQuantizationSubGraph(unittest.TestCase):

    def build_graph_with_sub_graph(self):
        if False:
            return 10

        def linear_fc(num):
            if False:
                print('Hello World!')
            data = paddle.static.data(name='image', shape=[-1, 1, 32, 32], dtype='float32')
            label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
            hidden = data
            for _ in range(num):
                hidden = paddle.static.nn.fc(hidden, size=128, activation='relu')
            loss = paddle.nn.functional.cross_entropy(input=hidden, label=label, reduction='none', use_softmax=False)
            loss = paddle.mean(loss)
            return loss
        main_program = Program()
        startup_program = Program()

        def true_func():
            if False:
                for i in range(10):
                    print('nop')
            return linear_fc(3)

        def false_func():
            if False:
                while True:
                    i = 10
            return linear_fc(5)
        with program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = paddle.less_than(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)
        core_graph = core.Graph(main_program.desc)
        graph = IrGraph(core_graph, for_test=True)
        sub_graph = graph.get_sub_graph(0)
        all_sub_graphs = graph.all_sub_graphs(for_test=True)
        return (graph, all_sub_graphs)

    def test_quant_sub_graphs(self, use_cuda=False):
        if False:
            i = 10
            return i + 15
        (graph, sub_graphs) = self.build_graph_with_sub_graph()
        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        transform_pass = QuantizationTransformPass(scope=base.global_scope(), place=place, activation_quantize_type='abs_max', weight_quantize_type='range_abs_max')
        Find_inserted_quant_op = False
        for sub_graph in sub_graphs:
            transform_pass.apply(sub_graph)
            for op in sub_graph.all_op_nodes():
                if 'quantize' in op.name():
                    Find_inserted_quant_op = True
        self.assertTrue(Find_inserted_quant_op)

    def test_quant_sub_graphs_cpu(self):
        if False:
            return 10
        self.test_quant_sub_graphs(use_cuda=False)

    @OpTestTool.skip_if(not paddle.is_compiled_with_cuda(), 'Not GPU version paddle')
    def test_quant_sub_graphs_gpu(self):
        if False:
            i = 10
            return i + 15
        self.test_quant_sub_graphs(use_cuda=True)
if __name__ == '__main__':
    unittest.main()
import contextlib
import sympy
import torch
import torch._inductor.config as inductor_config
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import SizeArg
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

class TestCodegenTriton(TorchTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()

        class DummyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x * 2
        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        if False:
            while True:
                i = 10
        self._stack.close()
        super().tearDown()

    @inductor_config.patch('triton.divisible_by_16', True)
    def test_config_of_sizearg(self):
        if False:
            return 10
        two = sympy.Integer(2)
        eight = sympy.Integer(8)
        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol('s0', positive=True, integer=True)
        s1 = sympy.Symbol('s1', positive=True, integer=True)
        self.assertEqual((2,), triton_utils.config_of([SizeArg('A', two), SizeArg('B', eight), SizeArg('C', sixteen), SizeArg('D', s0), SizeArg('E', s1)]).divisible_by_16)
        self.assertEqual((0, 2, 4, 5, 6), triton_utils.config_of([SizeArg('A', two * eight), SizeArg('B', eight * s0), SizeArg('C', two * eight * s0), SizeArg('D', s0 * s1), SizeArg('E', sixteen * s0), SizeArg('F', sixteen * eight * s0 * s1), SizeArg('G', two * eight * s0 * s1)]).divisible_by_16)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if HAS_CPU or HAS_CUDA:
        run_tests('sympy')
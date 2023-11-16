import os
import sys
import torch
from torch._C import parse_ir
from torch.testing import FileCheck
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestIgnorableArgs(JitTestCase):

    def test_slice_ignorable_args_for_slice(self):
        if False:
            for i in range(10):
                print('nop')
        graph_str = 'graph():\n            %13 : int = prim::Constant[value=0]()\n            %10 : bool = prim::Constant[value=0]()\n            %8 : NoneType = prim::Constant()\n            %0 : int = prim::Constant[value=1]()\n            %1 : int = prim::Constant[value=2]()\n            %2 : int = prim::Constant[value=3]()\n            %3 : int = prim::Constant[value=4]()\n            %4 : int = prim::Constant[value=9]()\n            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)\n            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)\n            %7 : int[][] = prim::ListConstruct(%5, %6)\n            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10)\n            %16 : Tensor = aten::slice(%val.1, %13, %1, %8, %0)\n            %20 : Tensor = aten::slice(%16, %0, %8, %0, %0)\n            return (%20)'
        graph = parse_ir(graph_str)
        function = self.createFunctionFromGraph(graph)
        function_copy = self.getExportImportCopy(function)
        src = str(function.code)
        FileCheck().check('torch.slice(torch.slice(torch.tensor(_0), 0, 2), 1, None, 1)').run(src)
        self.assertEqual(function(), function_copy())

    def test_add_out_ignorable_args(self):
        if False:
            return 10

        @torch.jit.script
        def fn(x: torch.Tensor, y: torch.Tensor):
            if False:
                return 10
            torch.add(x, y, out=y)
        FileCheck().check('torch.add(x, y, out=y)').run(fn.code)
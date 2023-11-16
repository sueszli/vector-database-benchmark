import os
import sys
import torch
from torch.testing import FileCheck
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestFunctionalBlocks(JitTestCase):

    def test_subgraph_creation(self):
        if False:
            return 10

        def fn(x, y, z):
            if False:
                i = 10
                return i + 15
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)
            z = z * z
            y = y * z
            if y < 2:
                y = y + 5
            return x + y + z
        graph = torch.jit.script(fn).graph
        self.run_pass('create_functional_graphs', graph)
        FileCheck().check('%x').check_not('%x').check('FunctionalGraph').check('%x').run(graph)
        FileCheck().check('%y').check_not('%y').check('FunctionalGraph').check('%y').run(graph)
        FileCheck().check('Tensor = prim::Functional').check_next('aten::add').run(graph)
        FileCheck().check('add').check('add_').check_not('mul').check('FunctionalGraph').run(graph)
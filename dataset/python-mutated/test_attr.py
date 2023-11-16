from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
import torch
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestGetDefaultAttr(JitTestCase):

    def test_getattr_with_default(self):
        if False:
            while True:
                i = 10

        class A(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.init_attr_val = 1.0

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                y = getattr(self, 'init_attr_val')
                w: list[float] = [1.0]
                z = getattr(self, 'missing', w)
                z.append(y)
                return z
        result = A().forward(0.0)
        self.assertEqual(2, len(result))
        graph = torch.jit.script(A()).graph
        FileCheck().check('prim::GetAttr[name="init_attr_val"]').run(graph)
        FileCheck().check_not('missing').run(graph)
        FileCheck().check('float[] = prim::ListConstruct').run(graph)
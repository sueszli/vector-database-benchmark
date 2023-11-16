import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase, make_global

class TestDCE(JitTestCase):

    def test_setattr_no_aliasdb(self):
        if False:
            for i in range(10):
                print('nop')

        class Net(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x = torch.empty([2, 2])

            def forward(self):
                if False:
                    while True:
                        i = 10
                x = torch.rand([3, 3])
                self.x = x
        net = torch.jit.script(Net())
        FileCheck().check('prim::SetAttr').run(net.graph)

    def test_setattr_removed(self):
        if False:
            return 10

        @torch.jit.script
        class Thing1:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = torch.zeros([2, 2])
        make_global(Thing1)

        class Thing2(torch.nn.Module):

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                x = torch.rand([2, 2])
                y = torch.rand([2, 2])
                t1 = Thing1()
                t1.x = x
                return y
        unscripted = Thing2()
        t2 = torch.jit.script(unscripted)
        t2.eval()
        t2 = torch.jit.freeze(t2)
        FileCheck().check_not('prim::SetAttr').run(t2.graph)
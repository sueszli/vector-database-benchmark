import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestEinsumOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'einsum'
        self.use_dynamic_create_class = False

    class TestEinsumBinary(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'einsum'
            self.disable = False
            self.types = [self.in_type, self.in_type]
            self.set_mandatory()
            self.init_input()
            np.random.seed(123)
            out = np.einsum(self.equation, *self.inputs)
            self.operands = []
            for (idx, inp) in enumerate(self.inputs):
                self.operands.append(('x' + str(idx), inp))
            self.inputs = {'Operands': self.operands}
            self.attrs = {'equation': self.equation}
            self.outputs = {'Out': out, 'InnerCache': [('cache_' + str(i), np.array([1.0])) for i in range(len(self.operands))], 'XShape': [('xshape_' + str(i), np.array([1.0])) for i in range(len(self.operands))]}

        def init_input(self):
            if False:
                for i in range(10):
                    print('nop')
            self.inputs = []
            for (t, s) in zip(self.types, self.shapes):
                self.inputs.append(np.random.random(s).astype(t))

        def set_mandatory(self):
            if False:
                while True:
                    i = 10
            self.shapes = [(10, 10, 20), (20, 6)]
            self.equation = 'mij,jk->ki'

        def test_check_output(self):
            if False:
                print('Hello World!')
            if not self.disable:
                self.check_output_with_place(paddle.XPUPlace(0), no_check_set=['InnerCache', 'XShape'], atol=0.005)

        def test_grad(self):
            if False:
                i = 10
                return i + 15
            if not self.disable:
                self.check_grad_with_place(paddle.XPUPlace(0), [op[0] for op in self.operands], ['Out'])

    class TestEinsum1(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(20, 3, 3), (20, 3, 3)]
            self.equation = 'mij,mjk->mik'

    class TestEinsum2(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(20, 3, 3), (20, 3, 3)]
            self.equation = 'mij,mjk->ikm'

    class TestEinsum3(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(10, 10), (10, 10)]
            self.equation = 'ij,jk->ik'

    class TestEinsumWithReduction(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shapes = [(10, 3, 5), (5, 30)]
            self.equation = 'ijk,kl->jl'

    class TestEinsumWithReduction1(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(10, 3, 3, 5), (10, 5, 10, 10)]
            self.equation = 'mijk,mklh->ljm'

    class TestEinsumWithUnary(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shapes = [(10, 10, 3, 5)]
            self.equation = 'mijk->mi'

    class TestEinsumWithUnary1(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
            self.equation = 'imjl,jklm->imk'

    class TestEinsumWithBroadcast1(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                while True:
                    i = 10
            self.shapes = [(5, 10, 3, 3)]
            self.equation = 'i...->...'

    class TestEinsumWithBroadcast2(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(10, 11), (3, 4, 5, 10)]
            self.equation = '...ij,...i->j...'

    class TestEinsumWithBroadcast4(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                return 10
            self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
            self.equation = 'a...d,...cb->...abcd'

    class TestEinsumWithBroadcast5(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                return 10
            self.shapes = [(3, 2, 2, 10), (10, 3, 2, 2)]
            self.equation = '...a,a...->...'

    class TestEinsumWithBroadcast6(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                return 10
            self.shapes = [100, 100]
            self.equation = 'i,i->'

    class TestEinsumWithDiagonal(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                while True:
                    i = 10
            self.shapes = [(10, 10)]
            self.equation = 'ii->'

    class TestEinsumWithDiagonal2(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shapes = [(10, 3, 10)]
            self.equation = 'iji->j'

    class TestEinsumWithDiagonal3(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                for i in range(10):
                    print('nop')
            self.shapes = [(5, 3, 2, 1, 4, 5)]
            self.equation = 'a...a->...'

    class TestEinsumWithDiagonal4(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                i = 10
                return i + 15
            self.shapes = [(5, 3, 2, 1, 4, 5)]
            self.equation = 'a...a->a...'

    class TestEinsumWithDiagonal5(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                return 10
            self.shapes = [(8, 8, 8)]
            self.equation = 'aaa->a'

    class TestEinsumWithDiagonal6(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                while True:
                    i = 10
            self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
            self.equation = 'ijki,jkjk->ik'

    class TestEinsumWithDiagonal8(TestEinsumBinary):

        def set_mandatory(self):
            if False:
                while True:
                    i = 10
            self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
            self.equation = 'ijki,jkjk->'
support_types = get_xpu_op_support_types('einsum')
for stype in support_types:
    create_test_class(globals(), XPUTestEinsumOp, stype)
if __name__ == '__main__':
    unittest.main()
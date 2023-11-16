import re
import sys
from io import StringIO
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.comptime import comptime
FILE = None
SELF = None

class ComptimeTests(torch._dynamo.test_case.TestCase):

    def test_print_graph(self):
        if False:
            for i in range(10):
                print('nop')
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                print('Hello World!')
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    print('Hello World!')
                ctx.print_graph(verbose=False, file=FILE)
            comptime.print_graph()
            return y + 3
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(FILE.getvalue().strip(), 'def forward(self, L_x_ : torch.Tensor):\n    l_x_ = L_x_\n    y = l_x_ * 2;  l_x_ = None')

    def test_print_disas(self):
        if False:
            print('Hello World!')
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                return 10
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    return 10
                ctx.print_disas(file=FILE)
            comptime.print_disas()
            return y + 3

        def munge_disas(s):
            if False:
                i = 10
                return i + 15
            re.sub('^(?: +\\d+)?(?: +(-->)) \\+\\d+ ([A-Za-z0-9_]+)', '\x01 \x03', s, flags=re.MULTILINE)
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        out = FILE.getvalue()
        self.assertIn('-->', out)
        self.assertIn('STORE_FAST', out)
        if sys.version_info < (3, 11):
            self.assertIn('BINARY_MULTIPLY', out)
        else:
            self.assertIn('BINARY_OP', out)

    def test_print_value_stack(self):
        if False:
            print('Hello World!')
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            if False:
                for i in range(10):
                    print('nop')

            @comptime
            def _(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.print_value_stack(file=FILE, stacklevel=1)
            return x

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                i = 10
                return i + 15
            y = x + g(x)
            return y + comptime.print_value_stack_and_return(y * 2)
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(FILE.getvalue(), '- TensorVariable()\n')

    def test_print_locals(self):
        if False:
            return 10
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                while True:
                    i = 10
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    print('Hello World!')
                ctx.print_locals(file=FILE)
            comptime.print_locals()
            return y + 3
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(FILE.getvalue(), 'x = TensorVariable()\ny = TensorVariable()\n')

    def test_print_bt(self):
        if False:
            return 10
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            if False:
                i = 10
                return i + 15

            @comptime
            def _(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.print_bt(file=FILE)
            comptime.print_bt()
            return x + 3

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                i = 10
                return i + 15
            y = x * 2
            y = g(y)
            return y + 3

        def munge_filenames(s):
            if False:
                print('Hello World!')
            return re.sub('File "[^"]+", line \\d+', 'File "X", line X', s)
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        bt = FILE.getvalue()
        self.assertIn('y = g(y)', bt)

    def test_print_guards(self):
        if False:
            return 10
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                print('Hello World!')
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    print('Hello World!')
                ctx.print_guards(file=FILE)
            comptime.print_guards()
            return y + 3
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(re.sub('\\s+$', '', FILE.getvalue().rstrip(), flags=re.MULTILINE), '\n        local "L[\'x\']" TENSOR_MATCH\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        global \'\' GRAD_MODE\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        global \'\' DETERMINISTIC_ALGORITHMS\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        global \'\' TORCH_FUNCTION_STATE\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        global \'\' DEFAULT_DEVICE\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        global \'\' BACKEND_MATCH\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }\n        shape_env \'\' SHAPE_ENV\n        {\n            \'guard_types\': None,\n            \'code\': None,\n            \'obj_weakref\': None\n            \'guarded_class\': None\n        }')

    def test_graph_break(self):
        if False:
            for i in range(10):
                print('nop')
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                print('Hello World!')
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    print('Hello World!')
                pass
            return y + 3
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        cnt.frame_count = 0

        @torch._dynamo.optimize(cnt)
        def g(x):
            if False:
                return 10
            y = x * 2

            @comptime
            def _(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.graph_break()
            y = y + 2
            comptime.graph_break()
            return y * 3
        g(torch.randn(2))
        self.assertEqual(cnt.frame_count, 3)

    def test_get_local(self):
        if False:
            print('Hello World!')
        global SELF, FILE
        SELF = self
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.optimize(cnt)
        def f(x):
            if False:
                while True:
                    i = 10
            y = x * 2
            lit = 2

            @comptime
            def _(ctx):
                if False:
                    i = 10
                    return i + 15
                y = ctx.get_local('y')
                SELF.assertEqual(y.as_fake().size(0), 2)
                SELF.assertEqual(y.size(0), 2)
                y.as_proxy() + 4
                ctx.print_graph(verbose=False, file=FILE)
                SELF.assertIs(y.python_type(), torch.Tensor)
                lit = ctx.get_local('lit')
                SELF.assertEqual(lit.as_python_constant(), 2)
            return y + 3
        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(FILE.getvalue().strip(), 'def forward(self, L_x_ : torch.Tensor):\n    l_x_ = L_x_\n    y = l_x_ * 2;  l_x_ = None\n    add = y + 4;  y = None')
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()
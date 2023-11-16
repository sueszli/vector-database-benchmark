import os
import sys
from textwrap import dedent
import torch
from torch.testing._internal import jit_utils
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestJitUtils(JitTestCase):

    def test_get_callable_argument_names_positional_or_keyword(self):
        if False:
            return 10

        def fn_positional_or_keyword_args_only(x, y):
            if False:
                print('Hello World!')
            return x + y
        self.assertEqual(['x', 'y'], torch._jit_internal.get_callable_argument_names(fn_positional_or_keyword_args_only))

    def test_get_callable_argument_names_positional_only(self):
        if False:
            return 10
        code = dedent('\n            def fn_positional_only_arg(x, /, y):\n                return x + y\n        ')
        fn_positional_only_arg = jit_utils._get_py3_code(code, 'fn_positional_only_arg')
        self.assertEqual(['y'], torch._jit_internal.get_callable_argument_names(fn_positional_only_arg))

    def test_get_callable_argument_names_var_positional(self):
        if False:
            for i in range(10):
                print('nop')

        def fn_var_positional_arg(x, *arg):
            if False:
                while True:
                    i = 10
            return x + arg[0]
        self.assertEqual(['x'], torch._jit_internal.get_callable_argument_names(fn_var_positional_arg))

    def test_get_callable_argument_names_keyword_only(self):
        if False:
            i = 10
            return i + 15

        def fn_keyword_only_arg(x, *, y):
            if False:
                print('Hello World!')
            return x + y
        self.assertEqual(['x'], torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg))

    def test_get_callable_argument_names_var_keyword(self):
        if False:
            print('Hello World!')

        def fn_var_keyword_arg(**args):
            if False:
                return 10
            return args['x'] + args['y']
        self.assertEqual([], torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg))

    def test_get_callable_argument_names_hybrid(self):
        if False:
            for i in range(10):
                print('nop')
        code = dedent("\n            def fn_hybrid_args(x, /, y, *args, **kwargs):\n                return x + y + args[0] + kwargs['z']\n        ")
        fn_hybrid_args = jit_utils._get_py3_code(code, 'fn_hybrid_args')
        self.assertEqual(['y'], torch._jit_internal.get_callable_argument_names(fn_hybrid_args))

    def test_checkscriptassertraisesregex(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                print('Hello World!')
            tup = (1, 2)
            return tup[2]
        self.checkScriptRaisesRegex(fn, (), Exception, 'range', name='fn')
        s = dedent('\n        def fn():\n            tup = (1, 2)\n            return tup[2]\n        ')
        self.checkScriptRaisesRegex(s, (), Exception, 'range', name='fn')

    def test_no_tracer_warn_context_manager(self):
        if False:
            i = 10
            return i + 15
        torch._C._jit_set_tracer_state_warn(True)
        with jit_utils.NoTracerWarnContextManager() as no_warn:
            self.assertEqual(False, torch._C._jit_get_tracer_state_warn())
        self.assertEqual(True, torch._C._jit_get_tracer_state_warn())
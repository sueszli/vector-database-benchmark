import os
import sys
import io
import torch
import warnings
from contextlib import redirect_stderr
from torch.testing import FileCheck
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestWarn(JitTestCase):

    def test_warn(self):
        if False:
            return 10

        @torch.jit.script
        def fn():
            if False:
                i = 10
                return i + 15
            warnings.warn('I am warning you')
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=1, exactly=True).run(f.getvalue())

    def test_warn_only_once(self):
        if False:
            return 10

        @torch.jit.script
        def fn():
            if False:
                i = 10
                return i + 15
            for _ in range(10):
                warnings.warn('I am warning you')
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=1, exactly=True).run(f.getvalue())

    def test_warn_only_once_in_loop_func(self):
        if False:
            while True:
                i = 10

        def w():
            if False:
                return 10
            warnings.warn('I am warning you')

        @torch.jit.script
        def fn():
            if False:
                return 10
            for _ in range(10):
                w()
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=1, exactly=True).run(f.getvalue())

    def test_warn_once_per_func(self):
        if False:
            i = 10
            return i + 15

        def w1():
            if False:
                while True:
                    i = 10
            warnings.warn('I am warning you')

        def w2():
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn('I am warning you')

        @torch.jit.script
        def fn():
            if False:
                return 10
            w1()
            w2()
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=2, exactly=True).run(f.getvalue())

    def test_warn_once_per_func_in_loop(self):
        if False:
            return 10

        def w1():
            if False:
                i = 10
                return i + 15
            warnings.warn('I am warning you')

        def w2():
            if False:
                i = 10
                return i + 15
            warnings.warn('I am warning you')

        @torch.jit.script
        def fn():
            if False:
                while True:
                    i = 10
            for _ in range(10):
                w1()
                w2()
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=2, exactly=True).run(f.getvalue())

    def test_warn_multiple_calls_multiple_warnings(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def fn():
            if False:
                i = 10
                return i + 15
            warnings.warn('I am warning you')
        f = io.StringIO()
        with redirect_stderr(f):
            fn()
            fn()
        FileCheck().check_count(str='UserWarning: I am warning you', count=2, exactly=True).run(f.getvalue())

    def test_warn_multiple_calls_same_func_diff_stack(self):
        if False:
            for i in range(10):
                print('nop')

        def warn(caller: str):
            if False:
                print('Hello World!')
            warnings.warn('I am warning you from ' + caller)

        @torch.jit.script
        def foo():
            if False:
                return 10
            warn('foo')

        @torch.jit.script
        def bar():
            if False:
                i = 10
                return i + 15
            warn('bar')
        f = io.StringIO()
        with redirect_stderr(f):
            foo()
            bar()
        FileCheck().check_count(str='UserWarning: I am warning you from foo', count=1, exactly=True).check_count(str='UserWarning: I am warning you from bar', count=1, exactly=True).run(f.getvalue())
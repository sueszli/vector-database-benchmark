import os
import sys
from typing import Any, List
import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase, make_global
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestWith(JitTestCase):
    """
    A suite of tests for with statements.
    """

    def test_with_as(self):
        if False:
            i = 10
            return i + 15
        "\n        Check that with statements that use the 'as' keyword to bind expressions\n        to targets work as expected.\n        "

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                if False:
                    print('Hello World!')
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                if False:
                    print('Hello World!')
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any) -> bool:
                if False:
                    print('Hello World!')
                self.count.sub_(0.3)
                return True
        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            'Basic test with one with-statement.'
            c = Context(1)
            with c as mult:
                y = x + mult
            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            '\n            Test with a pass statement inside a with-statement. Although\n            the body of the with is empty, __enter__ and __exit__ should\n            still be called.\n            '
            c = Context(1)
            with c as mult:
                pass
            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            '\n            Test that returning early from inside a with-statement works\n            as expected.\n            '
            with c as mult:
                y = x + mult
                return y
            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            '\n            Test that conditionally returning early from inside a with-statement works\n            as expected.\n            '
            with c as mult:
                y = x + mult
                if mult > 0:
                    return y
            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test that breaking early from inside a with-statement works\n            as expected.\n            '
            with c as mult:
                for a in l:
                    if a == 0:
                        break
                    x += a * mult
            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            '\n            Test that using continue inside a with-statement works\n            as expected.\n            '
            with c as mult:
                for a in l:
                    if a == 0:
                        continue
                    x += a * mult
            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            '\n            Test two with-statements in a row.\n            '
            c = Context(1)
            with c as mult:
                y = x + mult
            with c as mult:
                y *= mult
            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test nested with-statements.\n            '
            c = Context(1)
            with c as m:
                with c as n:
                    y = x + n
                y *= m
            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test a with-statement with multiple with items.\n            '
            c = Context(1)
            d = Context(2)
            with c as m, d as n:
                y = x + (m + n)
            return y
        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]
        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_no_as(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check that with statements that do not use the 'as' keyword to bind expressions\n        to targets work as expected.\n        "

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                if False:
                    return 10
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                if False:
                    i = 10
                    return i + 15
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                if False:
                    print('Hello World!')
                self.count.sub_(0.3)
        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            'Basic test with one with-statement.'
            c = Context(1)
            with c:
                y = x + c.count
            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            '\n            Test with a pass statement inside a with-statement. Although\n            the body of the with is empty, __enter__ and __exit__ should\n            still be called.\n            '
            c = Context(1)
            with c:
                pass
            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                return 10
            '\n            Test that returning early from inside a with-statement works\n            as expected.\n            '
            with c:
                y = x + c.count
                return y
            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test that conditionally returning early from inside a with-statement works\n            as expected.\n            '
            with c:
                y = x + c.count
                if c.count > 0:
                    return y
            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            '\n            Test that breaking early from inside a with-statement works\n            as expected.\n            '
            with c:
                for a in l:
                    if a == 0:
                        break
                    x += a * c.count
            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            '\n            Test that using continue inside a with-statement works\n            as expected.\n            '
            with c:
                for a in l:
                    if a == 0:
                        continue
                    x += a * c.count
            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test two with-statements in a row.\n            '
            c = Context(1)
            with c:
                y = x + c.count
            with c:
                y *= c.count
            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test nested with-statements.\n            '
            c = Context(1)
            with c:
                with c:
                    y = x + c.count
                y *= c.count
            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test a with-statement with multiple with items.\n            '
            c = Context(1)
            d = Context(2)
            with c, d:
                y = x + (c.count + d.count)
            return y
        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]
        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that exceptions thrown in the bodies of with-statements are\n        handled correctly.\n        '

        @torch.jit.script
        class Context:
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                if False:
                    return 10
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                if False:
                    i = 10
                    return i + 15
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                if False:
                    while True:
                        i = 10
                self.count.sub_(0.3)
        make_global(Context)

        @torch.jit.script
        def method_that_raises() -> torch.Tensor:
            if False:
                return 10
            raise Exception('raised exception')

        @torch.jit.script
        def test_exception(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                i = 10
                return i + 15
            '\n            Test the case in which an exception is thrown while executing the body of a with-statement.\n            '
            with c as _:
                x += method_that_raises()
            return x

        @torch.jit.script
        def test_exception_nested(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            '\n            Test the case in which an exception is thrown while executing the body of a nested with-statement.\n            '
            with c as _:
                with c as _:
                    x += method_that_raises()
            return x

        @torch.jit.script
        def with_that_raises(c: Context) -> torch.Tensor:
            if False:
                return 10
            a = torch.tensor([1])
            with c as _:
                a += method_that_raises()
            return a

        @torch.jit.script
        def test_exception_fn_call(x: torch.Tensor, c: Context) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test the case in which an exception is thrown while there are active with-statements in two different\n            frames.\n            '
            with c as _:
                x += with_that_raises(c)
            return x
        c = Context(1)
        with self.assertRaisesRegexWithHighlight(Exception, 'raised exception', 'raise Exception("raised exception'):
            test_exception(torch.randn(2), c)
        self.assertEqual(c.count, 1)
        with self.assertRaisesRegexWithHighlight(Exception, 'raised exception', 'raise Exception("raised exception'):
            test_exception_nested(torch.randn(2), c)
        self.assertEqual(c.count, 1)
        with self.assertRaisesRegexWithHighlight(Exception, 'raised exception', 'raise Exception("raised exception'):
            test_exception_fn_call(torch.randn(2), c)
        self.assertEqual(c.count, 1)

    def test_with_errors(self):
        if False:
            return 10
        '\n        Check that errors related to with-statements are detected and reported correctly.\n        '

        @torch.jit.script
        class NoEnterNoExit:
            """
            This class is missing __enter__ and __exit__ methods.
            """

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.count = 1

        @torch.jit.script
        class BadEnter:
            """
            This class has an __enter__ method with an incorrect signature.
            """

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.count = 1

            def __enter__(self, incr: int):
                if False:
                    while True:
                        i = 10
                self.count += incr

            def __exit__(self, type: Any, value: Any, tb: Any):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        @torch.jit.script
        class BadExit:
            """
            This class has an __exit__ method with an incorrect signature.
            """

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.count = 1

            def __enter__(self):
                if False:
                    print('Hello World!')
                self.count += 1

            def __exit__(self, type: Any, value: Any):
                if False:
                    while True:
                        i = 10
                pass

        @torch.jit.script
        class ExitIncorrectTypes:
            """
            This class has an __exit__ method with unsupported argument types.
            """

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.count = 1

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                self.count += 1

            def __exit__(self, type: Any, value: int, tb: int):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        def test_no_enter_no_exit(x: torch.Tensor, cm: NoEnterNoExit) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            with cm as _:
                pass
            return x

        def test_bad_enter(x: torch.Tensor, cm: BadEnter) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            with cm as _:
                pass
            return x

        def test_bad_exit(x: torch.Tensor, cm: BadExit) -> torch.Tensor:
            if False:
                print('Hello World!')
            with cm as _:
                pass
            return x

        def test_exit_incorrect_types(x: torch.Tensor, cm: ExitIncorrectTypes) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            with cm as _:
                pass
            return x

        def test_enter_without_object():
            if False:
                for i in range(10):
                    print('nop')
            with 'not_object' as obj:
                pass
        test_tensor = torch.randn(5, dtype=torch.double)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'does not define __enter__ and __exit__ methods', 'cm'):
            self.checkScript(test_no_enter_no_exit, (test_tensor, NoEnterNoExit()))
        with self.assertRaisesRegexWithHighlight(RuntimeError, '__enter__ must have only one argument and one return value', 'cm'):
            self.checkScript(test_bad_enter, (test_tensor, BadEnter()))
        with self.assertRaisesRegexWithHighlight(RuntimeError, '__exit__ must have four arguments', 'cm'):
            self.checkScript(test_bad_exit, (test_tensor, BadExit()))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'argument 2 of __exit__ must have Any type', 'cm'):
            self.checkScript(test_exit_incorrect_types, (test_tensor, ExitIncorrectTypes()))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'must return an object', '"not_object"'):
            self.checkScript(test_enter_without_object, ())

    def test_with_no_grad(self):
        if False:
            return 10
        '\n        Check that torch.no_grad() works. Most of these are adapted from\n        corresponding tests for eager-mode no_grad.\n        '

        def test_no_grad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            with torch.no_grad():
                w = x + y
            return w
        s = torch.jit.script(test_no_grad)
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        w = s(x, y)
        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda : w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)

        def test_no_grad_assignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            with torch.no_grad():
                x[0] = y
            return x
        s = torch.jit.script(test_no_grad_assignment)
        z = torch.randn(5)
        w = s(x, z)
        self.assertTrue(w.requires_grad)
        self.assertIsNone(w.grad_fn)

        class NoGradModule(torch.nn.Module):

            @torch.jit.ignore
            def adder(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                w = x + y
                return w

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                with torch.no_grad():
                    w = self.adder(x, y)
                return w
        s = torch.jit.script(NoGradModule())
        w = s(x, y)
        self.assertFalse(w.requires_grad)

    @skipIfTorchDynamo('Torchdynamo cannot correctly handle profiler.profile calls')
    def test_with_record_function(self):
        if False:
            print('Hello World!')
        '\n        Check that torch.autograd.profiler.record_function context manager is\n        torchscriptable.\n        '

        def with_rf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            with torch.autograd.profiler.record_function('foo'):
                with torch.autograd.profiler.record_function('nested'):
                    a = x + y
            return a
        scripted = torch.jit.script(with_rf)
        (x, y) = (torch.ones(2), torch.ones(2))
        with torch.autograd.profiler.profile() as p:
            scripted(x, y)
        p.key_averages()
        function_events = p.function_events
        rf_events = [evt for evt in function_events if evt.name == 'foo']
        self.assertEqual(len(rf_events), 1)
        rf_event = rf_events[0]
        child_events = rf_event.cpu_children
        self.assertTrue('nested' in (child.name for child in child_events))
        nested_function_event = [evt for evt in function_events if evt.name == 'nested'][0]
        nested_child_events = nested_function_event.cpu_children
        self.assertTrue('aten::add' in (child.name for child in nested_child_events))
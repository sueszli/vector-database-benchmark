import logging
import unittest
import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import IS_FBCODE, munge_exc, TEST_Z3
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test

class ExcTests(LoggingTestCase):
    maxDiff = None

    def test_unsupported_real_stack(self):
        if False:
            for i in range(10):
                print('nop')

        def fn002(x):
            if False:
                for i in range(10):
                    print('nop')
            torch._dynamo.graph_break()

        def fn001(x):
            if False:
                i = 10
                return i + 15
            x = x + 1
            fn002(x)
        self.assertExpectedInlineMunged(Unsupported, lambda : torch.compile(fn001, backend='eager', fullgraph=True)(torch.randn(1)), '\'call_function graph_break in skip_files _dynamo/decorators.py, skipped according skipfiles.SKIP_DIRS\'\n\nfrom user code:\n   File "test_exc.py", line N, in fn001\n    fn002(x)\n  File "test_exc.py", line N, in fn002\n    torch._dynamo.graph_break()')

    @torch._dynamo.config.patch(verbose=True, suppress_errors=True)
    @make_logging_test()
    @unittest.skipIf(IS_FBCODE, 'stack trace slightly different in fbcode')
    def test_internal_error_suppress_errors(self, records):
        if False:
            for i in range(10):
                print('nop')

        def fn001(x):
            if False:
                while True:
                    i = 10

            def f(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                raise AssertionError()
            comptime(f)
        torch.compile(fn001, backend='eager')(torch.randn(1))
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(munge_exc(record.getMessage()), 'WON\'T CONVERT fn001 test_exc.py line N\n========== TorchDynamo Stack Trace ==========\nTraceback (most recent call last):\n  File "test_exc.py", line N, in f\n    raise AssertionError()\nAssertionError:\n\nfrom user code:\n   File "test_exc.py", line N, in fn001\n    comptime(f)\n\n\n========== The above exception occurred while processing the following code ==========\n\n  File "test_exc.py", line N, in test_internal_error_suppress_errors\n    torch.compile(fn001, backend="eager")(torch.randn(1))\n  File "test_exc.py", line N, in fn001\n    comptime(f)\n\n==========')

    @make_logging_test()
    def test_not_implemented_error(self, records):
        if False:
            for i in range(10):
                print('nop')

        def fn001(x):
            if False:
                print('Hello World!')

            def f(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                raise NotImplementedError()
            for i in range(3):
                comptime(f)
        torch.compile(fn001, backend='eager')(torch.randn(1))
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(munge_exc(record.getMessage()), 'WON\'T CONVERT fn001 test_exc.py line N\ndue to:\nTraceback (most recent call last):\n  File "test_exc.py", line N, in f\n    raise NotImplementedError()\ntorch._dynamo.exc.InternalTorchDynamoError:\n\nfrom user code:\n   File "test_exc.py", line N, in fn001\n    comptime(f)')

    @unittest.expectedFailure
    @torch._dynamo.config.patch(inject_BUILD_SET_unimplemented_TESTING_ONLY=True)
    @make_logging_test(dynamo=logging.DEBUG)
    def test_unsupported_error(self, records):
        if False:
            return 10

        def fn001(x):
            if False:
                print('Hello World!')
            return {1, 2}
        torch.compile(fn001, backend='eager')(torch.randn(1))
        self.getRecord(records, 'Graph break:')

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_internal_error_no_suppress(self):
        if False:
            print('Hello World!')

        def fn001(x):
            if False:
                print('Hello World!')

            def f(ctx):
                if False:
                    i = 10
                    return i + 15
                raise AssertionError()
            comptime(f)
        self.assertExpectedInlineMunged(AssertionError, lambda : torch.compile(fn001, backend='eager')(torch.randn(1)), '\n\nfrom user code:\n   File "test_exc.py", line N, in fn001\n    comptime(f)')

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log(self, records):
        if False:
            for i in range(10):
                print('nop')

        def fn002(x):
            if False:
                while True:
                    i = 10
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 1
            return x

        def fn001(x):
            if False:
                while True:
                    i = 10
            return fn002(x)
        torch.compile(fn001, backend='eager')(torch.randn(1))
        record = self.getRecord(records, 'Graph break:')
        self.assertExpectedInline(munge_exc(record.getMessage()), 'Graph break: \'call_function graph_break in skip_files _dynamo/decorators.py, skipped according skipfiles.SKIP_DIRS\' from user code at:\n  File "test_exc.py", line N, in fn001\n    return fn002(x)\n  File "test_exc.py", line N, in fn002\n    torch._dynamo.graph_break()\n')

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_backend_suppress_line(self):
        if False:
            print('Hello World!')

        def fn001(x):
            if False:
                return 10
            x = torch.relu(x)
            return x + 1
        self.assertExpectedInlineMunged(torch._dynamo.exc.BackendCompilerFailed, lambda : torch.compile(fn001, backend='relu_compile_error_TESTING_ONLY')(torch.randn(1)), "backend='relu_compile_error_TESTING_ONLY' raised:\nReluCompileError:")

    @skipIf(not TEST_Z3, 'z3 not installed')
    @torch._dynamo.config.patch(assume_static_by_default=False, suppress_errors=False)
    @torch.fx.experimental._config.patch(inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True, translation_validation=True, translation_validation_no_bisect=True)
    def test_trigger_on_error(self):
        if False:
            while True:
                i = 10
        from torch.fx.experimental.validator import ValidationException

        @torch.compile
        def fn(x, shape):
            if False:
                return 10
            return x.split(shape)
        self.assertExpectedInlineMunged(ValidationException, lambda : fn(torch.randn(20), (5, 10, 5)), "translation validation failed.\n\nModel:\n  ==> L['shape'][0]: 0\n  ==> L['shape'][1]: 0\n  ==> L['shape'][2]: 0\n  ==> L['x'].size()[0]: 3\n  ==> L['x'].storage_offset(): 0\n  ==> L['x'].stride()[0]: 1\n  ==> s0: 3\n  ==> s1: 0\n  ==> s2: 0\n  ==> s3: 0\n\nAssertions:\n  ==> (== 0 L['x'].storage_offset())\n  ==> (== 1 L['x'].stride()[0])\n  ==> (== L['shape'][0] s1)\n  ==> (== L['shape'][1] s2)\n  ==> (== L['shape'][2] s3)\n  ==> (== L['x'].size()[0] s0)\n  ==> (> s0 1)\n  ==> (True)\n\nTarget Expressions:\n  ==> (<= 0 s1)\n  ==> (<= 0 s2)\n  ==> (<= 0 s3)\n  ==> (<= 2 s0)\n  ==> (== 0 L['shape'][0])\n  ==> (== 0 L['shape'][1])\n  ==> (== 0 L['shape'][2])\n  ==> (== 0 L['x'].storage_offset())\n  ==> (== 0 s1)\n  ==> (== 0 s2)\n  ==> (== 0 s3)\n  ==> (== 1 L['x'].stride()[0])\n  ==> (== L['x'].size()[0] s0)\n  ==> (> s0 0)\n  ==> (>= 9223372036854775806 s0)\n  ==> (>= 9223372036854775806 s1)\n  ==> (>= 9223372036854775806 s2)\n  ==> (>= 9223372036854775806 s3)\n\nFailed Source Expressions:\n  ==> (!= 0 L['shape'][0])\n  ==> (!= 0 L['shape'][1])\n  ==> (!= 0 L['shape'][2])\n  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])")

    @skipIf(not TEST_Z3, 'z3 not installed')
    @torch._dynamo.config.patch(assume_static_by_default=False, suppress_errors=False)
    @torch.fx.experimental._config.patch(inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True, translation_validation=True)
    def test_trigger_bisect_on_error(self):
        if False:
            print('Hello World!')
        from torch.fx.experimental.validator import BisectValidationException

        @torch.compile
        def fn(x, shape):
            if False:
                for i in range(10):
                    print('nop')
            return x.split(shape)
        self.assertExpectedInlineMunged(BisectValidationException, lambda : fn(torch.randn(20), (5, 10, 5)), "translation validation failed when evaluating: Eq(s1 + s2 + s3, s0)\n\nFailure occurred while running node:\n    %split : [num_users=3] = call_method[target=split](args = (%l_x_, (%l_shape_0_, %l_shape_1_, %l_shape_2_)), kwargs = {})\n\nModel:\n  ==> L['shape'][0]: -9223372036854775807\n  ==> L['shape'][1]: -9223372036854775807\n  ==> L['shape'][2]: -9223372036854775807\n  ==> L['x'].size()[0]: 3\n  ==> L['x'].storage_offset(): 0\n  ==> L['x'].stride()[0]: 1\n  ==> s0: 3\n  ==> s1: -9223372036854775807\n  ==> s2: -9223372036854775807\n  ==> s3: -9223372036854775807\n\nAssertions:\n  ==> (== 0 L['x'].storage_offset())\n  ==> (== 1 L['x'].stride()[0])\n  ==> (== L['shape'][0] s1)\n  ==> (== L['shape'][1] s2)\n  ==> (== L['shape'][2] s3)\n  ==> (== L['x'].size()[0] s0)\n  ==> (> s0 1)\n\nTarget Expressions:\n  ==> (!= (+ s1 s2 s3) s0)\n  ==> (<= -9223372036854775808 s1)\n  ==> (<= -9223372036854775808 s2)\n  ==> (<= -9223372036854775808 s3)\n  ==> (<= 2 s0)\n  ==> (== 0 L['x'].storage_offset())\n  ==> (== 1 L['x'].stride()[0])\n  ==> (== L['shape'][0] s1)\n  ==> (== L['shape'][1] s2)\n  ==> (== L['shape'][2] s3)\n  ==> (== L['x'].size()[0] s0)\n  ==> (> s0 0)\n  ==> (>= 9223372036854775806 s0)\n  ==> (>= 9223372036854775807 s1)\n  ==> (>= 9223372036854775807 s2)\n  ==> (>= 9223372036854775807 s3)\n\nFailed Source Expressions:\n  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])")
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()
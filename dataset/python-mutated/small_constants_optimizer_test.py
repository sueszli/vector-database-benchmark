import contextlib
import timeit
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

@contextlib.contextmanager
def options(optimizer_options):
    if False:
        return 10
    old_opts = context.context().get_optimizer_experimental_options()
    context.context().set_optimizer_experimental_options(optimizer_options)
    try:
        yield
    finally:
        context.context().set_optimizer_experimental_options(old_opts)

class FunctionTest(test.TestCase):

    @test_util.run_v2_only
    def test_grappler_optimization(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def brancher(inp):
            if False:
                print('Hello World!')
            x = constant_op.constant(1)
            for _ in range(1000):
                if inp:
                    x = x + constant_op.constant(1)
                else:
                    x = x + constant_op.constant(2)
            return x

        @polymorphic_function.function
        def brancher_true():
            if False:
                while True:
                    i = 10
            left = constant_op.constant(True)
            x = constant_op.constant(1)
            for _ in range(1000):
                if left:
                    x = x + constant_op.constant(1)
                else:
                    x = x + constant_op.constant(2)
            return x
        x = constant_op.constant(True)
        self.assertEqual(brancher(x), brancher_true())
        benchmark = min(timeit.repeat(lambda : brancher(x), repeat=5, number=100))
        opt_benchmark = min(timeit.repeat(brancher_true, repeat=5, number=100))
        self.assertLess(opt_benchmark * 3, benchmark)

    @test_util.run_v2_only
    def test_small_constants_optimization_with_grappler(self):
        if False:
            print('Hello World!')

        def func(inp):
            if False:
                while True:
                    i = 10
            x = constant_op.constant(1)
            for _ in range(1000):
                if inp:
                    x = x + constant_op.constant(1)
                else:
                    x = x + constant_op.constant(2)
            return x
        brancher = polymorphic_function.function(func)
        brancher_opt = polymorphic_function.function(func, experimental_attributes={'runtime_constant_optimization': True})
        with ops.device_v2('CPU'):
            x = constant_op.constant(True)
        self.assertEqual(brancher(x), brancher_opt(x))
        benchmark = min(timeit.repeat(lambda : brancher(x), repeat=5, number=100))
        opt_benchmark = min(timeit.repeat(lambda : brancher_opt(x), repeat=5, number=100))
        self.assertLess(opt_benchmark * 2, benchmark)

    @test_util.run_v2_only
    @test_util.run_gpu_only
    def test_small_constants_optimization_disabled(self):
        if False:
            print('Hello World!')

        @polymorphic_function.function(experimental_attributes={'runtime_constant_optimization': True})
        def func(inp):
            if False:
                while True:
                    i = 10
            return inp
        x = constant_op.constant(True)
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Expecting boolean tensor to be on host when small_constants_optimizer is enabled.'):
            func(x)

    @test_util.run_v2_only
    def test_small_constants_optimization_invalid_input(self):
        if False:
            return 10

        @polymorphic_function.function(experimental_attributes={'runtime_constant_optimization': True})
        def func(inp):
            if False:
                while True:
                    i = 10
            return inp
        with ops.device_v2('CPU'):
            x = constant_op.constant([True, True])
        self.assertAllEqual(func(x), x)

    @test_util.run_v2_only
    def test_small_constants_optimization_without_grappler(self):
        if False:
            for i in range(10):
                print('nop')

        def func(inp):
            if False:
                while True:
                    i = 10
            x = constant_op.constant(1)
            for _ in range(1000):
                if inp:
                    x = x + constant_op.constant(1)
                else:
                    x = x + constant_op.constant(2)
            return x
        brancher = polymorphic_function.function(func)
        brancher_opt = polymorphic_function.function(func, experimental_attributes={'runtime_constant_optimization': True})
        with ops.device_v2('CPU'):
            x = constant_op.constant(True)
        self.assertEqual(brancher(x), brancher_opt(x))
        with options({'disable_meta_optimizer': True}):
            benchmark = min(timeit.repeat(lambda : brancher(x), repeat=5, number=100))
            opt_benchmark = min(timeit.repeat(lambda : brancher_opt(x), repeat=5, number=100))
        self.assertLess(opt_benchmark * 5, benchmark)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()
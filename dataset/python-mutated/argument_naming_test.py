from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class ArgumentNamingTests(test.TestCase, parameterized.TestCase):
    """Tests for recognizable export signatures from concrete functions."""

    def testBasic(self):
        if False:
            return 10

        @polymorphic_function.function
        def fn(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b, a * b)
        fn(array_ops.ones([]), array_ops.ones([]))
        fn_op = fn.get_concrete_function(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'a', b'b'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertLen(fn_op.graph.structured_outputs, 2)
        self.assertAllClose([3.0, 2.0], fn_op(constant_op.constant(1.0), constant_op.constant(2.0)))
        self.assertAllClose([3.0, 2.0], fn_op(a=constant_op.constant(1.0), b=constant_op.constant(2.0)))

    def testVariable(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def fn(a, b):
            if False:
                return 10
            return (a + b, a * b)
        fn(array_ops.ones([]), array_ops.ones([]))
        fn_op = fn.get_concrete_function(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32), variables.Variable(1.0))
        self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'a', b'b'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertLen(fn_op.graph.structured_outputs, 2)

    def testDictReturned(self):
        if False:
            print('Hello World!')

        @polymorphic_function.function
        def fn(x, z=(1.0, 2.0), y=3.0):
            if False:
                print('Hello World!')
            (z1, z2) = z
            return {'alpha': x + y + z1, 'beta': x * y + z2}
        fn(array_ops.ones([]))
        fn_op = fn.get_concrete_function(x=tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32), y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x', 'y'], [inp.op.name for inp in fn_op.inputs])
        self.assertEqual([b'x', b'y'], [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
        self.assertEqual({'alpha', 'beta'}, set(fn_op.graph.structured_outputs.keys()))
        fn_op2 = fn.get_concrete_function(z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32, name='z_first'), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z_second')), y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'), x=4.0)
        self.assertEqual(['z_first', 'z_second', 'custom'], [inp.op.name for inp in fn_op2.inputs])
        self.assertEqual([b'z_first', b'z_second', b'custom'], [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs])
        fn_op3 = fn.get_concrete_function(tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'), z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32, name='z1'), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z2')), y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['custom', 'z1', 'z2', 'y'], [inp.op.name for inp in fn_op3.inputs])
        self.assertEqual([b'custom', b'z1', b'z2', b'y'], [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs])

    def testMethod(self):
        if False:
            while True:
                i = 10

        class HasMethod(object):

            @polymorphic_function.function
            def method(self, x):
                if False:
                    print('Hello World!')
                return x
        has_method = HasMethod()
        HasMethod.method(has_method, array_ops.ones([]))
        class_op = HasMethod.method.get_concrete_function(has_method, tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x'], [inp.op.name for inp in class_op.inputs])
        self.assertEqual([b'x'], [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs])
        has_method.method(array_ops.ones([]))
        method_op = has_method.method.get_concrete_function(tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
        self.assertEqual(['x'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'x'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
        self.skipTest('Not working')
        method_op = has_method.method.get_concrete_function(tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='y'))
        self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])

    def testMethodSignature(self):
        if False:
            print('Hello World!')

        class HasMethod(object):

            @polymorphic_function.function(input_signature=(tensor_spec.TensorSpec(shape=None, dtype=dtypes.float64, name='y'),))
            def method(self, x):
                if False:
                    return 10
                hash(self)
                return x
        has_method = HasMethod()
        has_method.method(array_ops.ones([], dtype=dtypes.float64))
        method_op = has_method.method.get_concrete_function()
        self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
        method_op2 = has_method.method.get_concrete_function()
        self.assertEqual(['y'], [inp.op.name for inp in method_op2.inputs])
        self.assertEqual([b'y'], [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs])

    def testVariadic(self):
        if False:
            return 10

        @polymorphic_function.function
        def variadic_fn(x, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return x + math_ops.add_n(list(args) + list(kwargs.values()))
        variadic_fn(array_ops.ones([]), array_ops.ones([]))
        variadic_op = variadic_fn.get_concrete_function(tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32), tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='second_variadic'), z=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32), zz=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'))
        self.assertEqual(['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'], [inp.op.name for inp in variadic_op.inputs])
        self.assertEqual([b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'], [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])

    def testVariadicInputSignature(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function(input_signature=(tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32), tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32), tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z')))
        def variadic_fn(x, *args):
            if False:
                for i in range(10):
                    print('nop')
            return x + math_ops.add_n(list(args))
        variadic_fn(array_ops.ones([]), array_ops.ones([]), array_ops.ones([]), array_ops.ones([]))
        variadic_op = variadic_fn.get_concrete_function()
        self.assertIn(b'variadic_fn', variadic_op.name)
        self.assertEqual(['x', 'y', 'args_1', 'z'], [inp.op.name for inp in variadic_op.inputs])
        self.assertEqual([b'x', b'y', b'args_1', b'z'], [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])
if __name__ == '__main__':
    ops.enable_eager_execution(config=config_pb2.ConfigProto(device_count={'CPU': 4}))
    test.main()
"""Tests for operator dispatch."""
import collections
import typing
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.proto_ops import decode_proto
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export

class CustomTensor(object):
    """A fake composite tensor class, for testing type-based dispatching."""

    def __init__(self, tensor, score):
        if False:
            return 10
        self.tensor = ops.convert_to_tensor(tensor)
        self.score = score

@tf_export('test_op')
@dispatch.add_dispatch_support
def test_op(x, y, z):
    if False:
        i = 10
        return i + 15
    'A fake op for testing dispatch of Python ops.'
    return x + 2 * y + 3 * z

@tf_export('test_op_with_optional')
@dispatch.add_dispatch_support
def test_op_with_optional(x, y, z, optional=None):
    if False:
        return 10
    'A fake op for testing dispatch of Python ops.'
    del optional
    return x + 2 * y + 3 * z

@tf_export('test_op_with_kwonly')
@dispatch.add_dispatch_support
def test_op_with_kwonly(*, x, y, z, optional=None):
    if False:
        print('Hello World!')
    'A fake op for testing dispatch of Python ops.'
    del optional
    return x + 2 * y + 3 * z

class TensorTracer(object):
    """An object used to trace TensorFlow graphs.

  This is an example class that is used to test global op dispatchers.  The
  global op dispatcher for TensorTracers is defined below.
  """

    def __init__(self, name, args=None, kwargs=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.shape = array_ops.ones(shape=(4, 4)).shape
        self.dtype = dtypes.float32

    def __repr__(self):
        if False:
            return 10
        if self.args is None and self.kwargs is None:
            return self.name
        else:
            args = [str(x) for x in self.args]
            args += sorted(['{}={}'.format(name, x) for (name, x) in self.kwargs.items()])
            return '{}({})'.format(self.name, ', '.join(args))

    @property
    def is_tensor_like(self):
        if False:
            while True:
                i = 10
        return True

    @classmethod
    def _overload_all_operators(cls):
        if False:
            print('Hello World!')
        'Register overloads for all operators.'
        for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
            cls._overload_operator(operator)

    @classmethod
    def _overload_operator(cls, operator):
        if False:
            for i in range(10):
                print('nop')
        'Overload an operator with the same overloading as `tensor_lib.Tensor`.'
        tensor_oper = getattr(tensor_lib.Tensor, operator)
        tensor_oper = getattr(tensor_oper, '__func__', tensor_oper)
        setattr(cls, operator, tensor_oper)
TensorTracer._overload_all_operators()

class TensorTracerOpDispatcher(dispatch.GlobalOpDispatcher):
    """Global op dispatcher for TensorTracer."""

    def _flatten_with_slice_flattening(self, x):
        if False:
            return 10
        flat = []
        for val in nest.flatten(x):
            if isinstance(val, slice):
                flat.extend((val.start, val.stop, val.step))
            else:
                flat.append(val)
        return flat

    def handle(self, op, args, kwargs):
        if False:
            i = 10
            return i + 15
        if not (any((self.is_tensor_tracer_arg(x) for x in args)) or any((self.is_tensor_tracer_arg(x) for x in kwargs.values()))):
            return self.NOT_SUPPORTED
        symbol_name = get_canonical_name_for_symbol(op)
        return TensorTracer(symbol_name, args, kwargs)

    def is_tensor_tracer_arg(self, value):
        if False:
            for i in range(10):
                print('nop')
        return any((isinstance(x, TensorTracer) for x in self._flatten_with_slice_flattening(value)))

@test_util.run_all_in_graph_and_eager_modes
class DispatchTest(test_util.TensorFlowTestCase):

    def testAddDispatchForTypes_With_CppOp(self):
        if False:
            print('Hello World!')
        original_handlers = gen_math_ops.atan2._tf_fallback_dispatchers[:]

        @dispatch.dispatch_for_types(gen_math_ops.atan2, CustomTensor)
        def custom_atan2(y, x, name=None):
            if False:
                while True:
                    i = 10
            return CustomTensor(gen_math_ops.add(y.tensor, x.tensor, name), (x.score + y.score) / 2.0)
        self.assertEqual(len(math_ops.atan2._tf_fallback_dispatchers), len(original_handlers) + 1)
        x = CustomTensor([1.0, 2.0, 3.0], 2.0)
        y = CustomTensor([7.0, 8.0, 2.0], 0.0)
        x_plus_y = gen_math_ops.atan2(y, x)
        self.assertAllEqual(self.evaluate(x_plus_y.tensor), [8, 10, 5])
        self.assertNear(x_plus_y.score, 1.0, 0.001)
        a = [1.0, 2.0, 3.0]
        b = [7.0, 8.0, 2.0]
        a_plus_b = gen_math_ops.atan2(a, b)
        self.assertAllClose(a_plus_b, [0.14189707, 0.24497867, 0.98279375])
        with self.assertRaises((TypeError, ValueError)):
            gen_math_ops.atan2(a, None)
        gen_math_ops.atan2._tf_fallback_dispatchers = original_handlers

    def testAddDispatchForTypes_With_PythonOp(self):
        if False:
            i = 10
            return i + 15
        original_handlers = test_op._tf_fallback_dispatchers[:]

        def override_for_test_op(x, y, z):
            if False:
                return 10
            return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)
        override = dispatch.dispatch_for_types(test_op, CustomTensor)(override_for_test_op)
        self.assertIs(override, override_for_test_op)
        x = CustomTensor([1, 2, 3], 0.2)
        y = CustomTensor([7, 8, 2], 0.4)
        z = CustomTensor([0, 1, 2], 0.6)
        result = test_op(x, y, z)
        self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
        self.assertNear(result.score, 0.4, 0.001)
        test_op._tf_fallback_dispatchers = original_handlers

    def testDispatchForTypes_MissingArgs(self):
        if False:
            while True:
                i = 10
        original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

        def override_for_test_op(x, y, z):
            if False:
                i = 10
                return i + 15
            return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)
        override = dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)(override_for_test_op)
        self.assertIs(override, override_for_test_op)
        x = CustomTensor([1, 2, 3], 0.2)
        y = CustomTensor([7, 8, 2], 0.4)
        z = CustomTensor([0, 1, 2], 0.6)
        result = test_op_with_optional(x, y, z)
        self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
        self.assertNear(result.score, 0.4, 0.001)
        test_op_with_optional._tf_fallback_dispatchers = original_handlers

    def testDispatchForTypes_ProvidingMissingArgs(self):
        if False:
            while True:
                i = 10
        original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

        @dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)
        def override_for_test_op(x, y, z):
            if False:
                print('Hello World!')
            return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)
        x = CustomTensor([1, 2, 3], 0.2)
        y = CustomTensor([7, 8, 2], 0.4)
        z = CustomTensor([0, 1, 2], 0.6)
        with self.assertRaisesRegex(AssertionError, 'Dispatched op is called with argument `optional` set to a non-default value, which is not supported by the decorated function'):
            test_op_with_optional(x, y, z, optional=3)
        test_op_with_optional._tf_fallback_dispatchers = original_handlers

    def testDispatchForTypes_NewArgs(self):
        if False:
            for i in range(10):
                print('nop')
        original_handlers = test_op_with_optional._tf_fallback_dispatchers[:]

        @dispatch.dispatch_for_types(test_op_with_optional, CustomTensor)
        def override_for_test_op(x, y, z, u=None):
            if False:
                print('Hello World!')
            del u
            return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)
        x = CustomTensor([1, 2, 3], 0.2)
        y = CustomTensor([7, 8, 2], 0.4)
        z = CustomTensor([0, 1, 2], 0.6)
        result = test_op_with_optional(x, y, z)
        self.assertAllEqual(self.evaluate(result.tensor), [15, 21, 13])
        self.assertNear(result.score, 0.4, 0.001)
        test_op_with_optional._tf_fallback_dispatchers = original_handlers

    def testDispatchForTypes_SignatureMismatchOrder(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(AssertionError, "The decorated function's non-default arguments must be identical to that of the overridden op."):

            @dispatch.dispatch_for_types(test_op, CustomTensor)
            def override_for_test_op(x, z, y):
                if False:
                    i = 10
                    return i + 15
                return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)

    def testDispatchForTypes_MissingKwOnly(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(AssertionError, "The decorated function's non-default arguments must be identical to that of the overridden op."):

            @dispatch.dispatch_for_types(test_op_with_kwonly, CustomTensor)
            def override_for_test_op(x, z, y):
                if False:
                    while True:
                        i = 10
                return CustomTensor(test_op(x.tensor, y.tensor, z.tensor), (x.score + y.score + z.score) / 3.0)

    def testDispatchForTypes_SignatureMismatchNames(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(AssertionError, "The decorated function's non-default arguments must be identical to that of the overridden op."):

            @dispatch.dispatch_for_types(test_op, CustomTensor)
            def override_for_test_op(a, b, c):
                if False:
                    for i in range(10):
                        print('nop')
                return CustomTensor(test_op(a.tensor, b.tensor, c.tensor), (a.score + b.score + c.score) / 3.0)

    def testDispatchForTypes_OpDoesNotSupportDispatch(self):
        if False:
            print('Hello World!')

        def some_op(x, y):
            if False:
                return 10
            return x + y
        with self.assertRaisesRegex(AssertionError, 'Dispatching not enabled for'):

            @dispatch.dispatch_for_types(some_op, CustomTensor)
            def override_for_some_op(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x if x.score > 0 else y

    @test.mock.patch.object(tf_logging, 'warning', autospec=True)
    def testInteractionWithDeprecationWarning(self, mock_warning):
        if False:
            while True:
                i = 10

        @deprecation.deprecated(date=None, instructions='Instructions')
        @dispatch.add_dispatch_support
        def some_op(x):
            if False:
                i = 10
                return i + 15
            return x
        some_op(5)
        message = mock_warning.call_args[0][0] % mock_warning.call_args[0][1:]
        self.assertRegex(message, '.*some_op \\(from __main__\\) is deprecated and will be removed in a future version.*')

    def testGlobalDispatcher(self):
        if False:
            for i in range(10):
                print('nop')
        original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
        try:
            TensorTracerOpDispatcher().register()
            x = TensorTracer('x')
            y = TensorTracer('y')
            trace = math_ops.reduce_sum(math_ops.add(math_ops.abs(x), y), axis=3)
            self.assertEqual(str(trace), 'math.reduce_sum(math.add(math.abs(x), y), axis=3)')
            proto_val = TensorTracer('proto')
            trace = decode_proto(proto_val, 'message_type', ['field'], ['float32'])
            self.assertIn('io.decode_proto(bytes=proto,', str(trace))
        finally:
            dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

    def testGlobalDispatcherConvertToTensor(self):
        if False:
            return 10
        original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
        try:
            TensorTracerOpDispatcher().register()
            x = TensorTracer('x')
            y = TensorTracer('y')
            trace = math_ops.add(math_ops.abs(tensor_conversion.convert_to_tensor_v2_with_dispatch(x)), y)
            self.assertEqual(str(trace), 'math.add(math.abs(convert_to_tensor(x)), y)')
        finally:
            dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

    def testGlobalDispatcherGetItem(self):
        if False:
            for i in range(10):
                print('nop')
        original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
        try:
            TensorTracerOpDispatcher().register()
            x = TensorTracer('x')
            trace = x[0]
            self.assertEqual(str(trace), '__operators__.getitem(x, 0)')
            x = TensorTracer('x')
            y = TensorTracer('y')
            trace = x[y]
            self.assertEqual(str(trace), '__operators__.getitem(x, y)')
            x = TensorTracer('x')
            y = TensorTracer('y')
            trace = x[:y]
            self.assertEqual(str(trace), '__operators__.getitem(x, slice(None, y, None))')
            x = array_ops.ones(shape=(3, 3))
            y = TensorTracer('y')
            trace = x[y]
            self.assertEqual(str(trace), '__operators__.getitem(%s, y)' % x)
            trace = x[:y]
            self.assertEqual(str(trace), '__operators__.getitem(%s, slice(None, y, None))' % x)
        finally:
            dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

    def testGlobalDispatcherLinearOperators(self):
        if False:
            while True:
                i = 10
        original_global_dispatchers = dispatch._GLOBAL_DISPATCHERS
        try:
            TensorTracerOpDispatcher().register()
            x = TensorTracer('x')
            trace = linear_operator_diag.LinearOperatorDiag(x).eigvals()
            self.assertEqual(str(trace), 'convert_to_tensor(convert_to_tensor(x, dtype=None, dtype_hint=None, name=diag))')
            trace = linear_operator_diag.LinearOperatorDiag(x).add_to_tensor(x)
            self.assertIn('linalg.set_diag(convert_to_tensor(x, name=x), __operators__.add(convert_to_tensor(x, dtype=None, dtype_hint=None, name=diag), linalg.diag_part(convert_to_tensor(x, name=x)), name=', str(trace))
            trace = linear_operator_diag.LinearOperatorDiag(x).assert_non_singular()
            self.assertIn('debugging.assert_less', str(trace))
            self.assertIn('message=Singular operator:  Diagonal contained zero values.', str(trace))
        finally:
            dispatch._GLOBAL_DISPATCHERS = original_global_dispatchers

class MaskedTensor(extension_type.ExtensionType):
    """Simple ExtensionType for testing v2 dispatch."""
    values: tensor_lib.Tensor
    mask: tensor_lib.Tensor

class SillyTensor(extension_type.ExtensionType):
    """Simple ExtensionType for testing v2 dispatch."""
    value: tensor_lib.Tensor
    how_silly: float

@test_util.run_all_in_graph_and_eager_modes
class DispatchV2Test(test_util.TensorFlowTestCase):

    def testDispatchForOneSignature(self):
        if False:
            i = 10
            return i + 15

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor, 'y': MaskedTensor})
        def masked_add(x, y, name=None):
            if False:
                print('Hello World!')
            with ops.name_scope(name):
                return MaskedTensor(x.values + y.values, x.mask & y.mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            z = math_ops.add(x, y)
            self.assertAllEqual(z.values, x.values + y.values)
            self.assertAllEqual(z.mask, x.mask & y.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchSignatureWithUnspecifiedParameter(self):
        if False:
            return 10

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor})
        def masked_add(x, y):
            if False:
                for i in range(10):
                    print('nop')
            if y is None:
                return x
            y_values = y.values if isinstance(y, MaskedTensor) else y
            y_mask = y.mask if isinstance(y, MaskedTensor) else True
            return MaskedTensor(x.values + y_values, x.mask & y_mask)
        try:
            a = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            b = constant_op.constant([10, 20, 30, 40, 50])
            c = [10, 20, 30, 40, 50]
            d = 50
            e = None
            self.assertAllEqual(math_ops.add(a, b).values, [11, 22, 33, 44, 55])
            self.assertAllEqual(math_ops.add(a, c).values, [11, 22, 33, 44, 55])
            self.assertAllEqual(math_ops.add(a, d).values, [51, 52, 53, 54, 55])
            self.assertAllEqual(math_ops.add(a, e).values, [1, 2, 3, 4, 5])
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchForMultipleSignatures(self):
        if False:
            print('Hello World!')

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor}, {'y': MaskedTensor})
        def masked_add(x, y, name=None):
            if False:
                return 10
            with ops.name_scope(name):
                x_values = x.values if isinstance(x, MaskedTensor) else x
                x_mask = x.mask if isinstance(x, MaskedTensor) else True
                y_values = y.values if isinstance(y, MaskedTensor) else y
                y_mask = y.mask if isinstance(y, MaskedTensor) else True
                return MaskedTensor(x_values + y_values, x_mask & y_mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = constant_op.constant([10, 20, 30, 40, 50])
            z = math_ops.add(x, y)
            self.assertAllEqual(z.values, x.values + y)
            self.assertAllEqual(z.mask, x.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchForList(self):
        if False:
            print('Hello World!')

        @dispatch.dispatch_for_api(array_ops.concat, {'values': typing.List[MaskedTensor]})
        def masked_concat(values, axis, name=None):
            if False:
                return 10
            with ops.name_scope(name):
                return MaskedTensor(array_ops.concat([v.values for v in values], axis), array_ops.concat([v.mask for v in values], axis))
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1], [1, 1, 0])
            z = array_ops.concat([x, y], axis=0)
            self.assertAllEqual(z.values, array_ops.concat([x.values, y.values], 0))
            self.assertAllEqual(z.mask, array_ops.concat([x.mask, y.mask], 0))
        finally:
            dispatch.unregister_dispatch_for(masked_concat)

    def testDispatchForUnion(self):
        if False:
            print('Hello World!')
        MaybeMasked = typing.Union[MaskedTensor, tensor_lib.Tensor]

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaybeMasked, 'y': MaybeMasked})
        def masked_add(x, y, name=None):
            if False:
                return 10
            with ops.name_scope(name):
                x_values = x.values if isinstance(x, MaskedTensor) else x
                x_mask = x.mask if isinstance(x, MaskedTensor) else True
                y_values = y.values if isinstance(y, MaskedTensor) else y
                y_mask = y.mask if isinstance(y, MaskedTensor) else True
                return MaskedTensor(x_values + y_values, x_mask & y_mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = constant_op.constant([10, 20, 30, 40, 50])
            z = math_ops.add(x, y)
            self.assertAllEqual(z.values, x.values + y)
            self.assertAllEqual(z.mask, x.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchForTensorLike(self):
        if False:
            while True:
                i = 10
        MaskedOrTensorLike = typing.Union[MaskedTensor, core_tf_types.TensorLike]

        @dispatch.dispatch_for_api(math_ops.add)
        def masked_add(x: MaskedOrTensorLike, y: MaskedOrTensorLike, name=None):
            if False:
                print('Hello World!')
            with ops.name_scope(name):
                x_values = x.values if isinstance(x, MaskedTensor) else x
                x_mask = x.mask if isinstance(x, MaskedTensor) else True
                y_values = y.values if isinstance(y, MaskedTensor) else y
                y_mask = y.mask if isinstance(y, MaskedTensor) else True
                return MaskedTensor(x_values + y_values, x_mask & y_mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y1 = [10, 20, 30, 40, 50]
            y2 = np.array([10, 20, 30, 40, 50])
            y3 = constant_op.constant([10, 20, 30, 40, 50])
            y4 = variables.Variable([5, 4, 3, 2, 1])
            if not context.executing_eagerly():
                self.evaluate(variables.global_variables_initializer())
            for y in [y1, y2, y3, y4]:
                z = math_ops.add(x, y)
                self.assertAllEqual(z.values, x.values + y)
                self.assertAllEqual(z.mask, x.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchForOptional(self):
        if False:
            for i in range(10):
                print('nop')

        @dispatch.dispatch_for_api(array_ops.where_v2, {'condition': MaskedTensor, 'x': typing.Optional[MaskedTensor], 'y': typing.Optional[MaskedTensor]})
        def masked_where(condition, x=None, y=None, name=None):
            if False:
                print('Hello World!')
            del condition, x, y, name
            return 'stub'
        try:
            x = MaskedTensor([True, False, True, True, True], [1, 0, 1, 1, 1])
            self.assertEqual(array_ops.where_v2(x), 'stub')
            self.assertEqual(array_ops.where_v2(x, x, x), 'stub')
        finally:
            dispatch.unregister_dispatch_for(masked_where)

    def testDispatchForSignatureFromAnnotations(self):
        if False:
            i = 10
            return i + 15

        @dispatch.dispatch_for_api(math_ops.add)
        def masked_add(x: MaskedTensor, y: MaskedTensor, name=None):
            if False:
                while True:
                    i = 10
            with ops.name_scope(name):
                return MaskedTensor(x.values + y.values, x.mask & y.mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            z = math_ops.add(x, y)
            self.assertAllEqual(z.values, x.values + y.values)
            self.assertAllEqual(z.mask, x.mask & y.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchForPositionalSignature(self):
        if False:
            return 10

        @dispatch.dispatch_for_api(math_ops.add, {0: MaskedTensor, 1: MaskedTensor})
        def masked_add(x, y, name=None):
            if False:
                print('Hello World!')
            with ops.name_scope(name):
                return MaskedTensor(x.values + y.values, x.mask & y.mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            z = math_ops.add(x, y)
            self.assertAllEqual(z.values, x.values + y.values)
            self.assertAllEqual(z.mask, x.mask & y.mask)
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchWithVarargs(self):
        if False:
            for i in range(10):
                print('nop')

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor, 'y': MaskedTensor})
        def masked_add(*args, **kwargs):
            if False:
                print('Hello World!')
            self.assertAllEqual(args[0].values, x.values)
            self.assertAllEqual(args[1].values, y.values)
            self.assertEmpty(kwargs)
            return 'stub'
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            self.assertEqual(math_ops.add(x, y), 'stub')
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchWithKwargs(self):
        if False:
            for i in range(10):
                print('nop')

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor, 'y': MaskedTensor})
        def masked_add(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            self.assertAllEqual(kwargs['x'].values, x.values)
            self.assertAllEqual(kwargs['y'].values, y.values)
            self.assertEmpty(args)
            return 'stub'
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            self.assertEqual(math_ops.add(x=x, y=y), 'stub')
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchErrorForBadAPI(self):
        if False:
            print('Hello World!')

        def api_without_dispatch_support(x):
            if False:
                return 10
            return x + 1
        with self.assertRaisesRegex(ValueError, '.* does not support dispatch.'):

            @dispatch.dispatch_for_api(api_without_dispatch_support, {'x': MaskedTensor})
            def my_version(x):
                if False:
                    return 10
                del x

    def testDispatchErrorForNoSignature(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'must be called with at least one signature'):

            @dispatch.dispatch_for_api(math_ops.add)
            def my_add(x, y, name=None):
                if False:
                    while True:
                        i = 10
                del x, y, name

    def testDispatchErrorSignatureMismatchParamName(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, "Dispatch function's signature \\(x, why, name=None\\) does not match API's signature \\(x, y, name=None\\)."):

            @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor})
            def my_add(x, why, name=None):
                if False:
                    while True:
                        i = 10
                del x, why, name

    def testDispatchErrorSignatureMismatchExtraParam(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, "Dispatch function's signature \\(x, y, name=None, extra_arg=None\\) does not match API's signature \\(x, y, name=None\\)."):

            @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor})
            def my_add(x, y, name=None, extra_arg=None):
                if False:
                    return 10
                del x, y, name, extra_arg

    def testDispatchErrorForUnsupportedTypeAnnotation(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'Type annotation .* is not currently supported by dispatch.'):

            @dispatch.dispatch_for_api(math_ops.add, {'x': typing.Tuple[MaskedTensor]})
            def my_add(x, y, name=None):
                if False:
                    while True:
                        i = 10
                del x, y, name

    def testDispatchErrorForUnknownParameter(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, "signature includes annotation for unknown parameter 'z'."):

            @dispatch.dispatch_for_api(math_ops.add, {'z': MaskedTensor})
            def my_add(x, y, name=None):
                if False:
                    while True:
                        i = 10
                del x, y, name

    def testDispatchErrorUnsupportedKeywordOnlyAnnotation(self):
        if False:
            i = 10
            return i + 15

        @dispatch.add_dispatch_support
        def foo(x, *, y):
            if False:
                return 10
            return x + y
        with self.assertRaisesRegex(ValueError, 'Dispatch currently only supports type annotations for positional parameters'):

            @dispatch.dispatch_for_api(foo, {'y': MaskedTensor})
            def masked_foo(x, *, y):
                if False:
                    print('Hello World!')
                del x, y

    def testDispatchErrorBadSignatureType(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'signatures must be dictionaries mapping parameter names to type annotations'):

            @dispatch.dispatch_for_api(math_ops.add, [MaskedTensor])
            def my_add(x, y, name=None):
                if False:
                    i = 10
                    return i + 15
                del x, y, name
        with self.assertRaisesRegex(TypeError, 'signatures must be dictionaries mapping parameter names to type annotations'):

            @dispatch.dispatch_for_api(math_ops.multiply, {None: MaskedTensor})
            def my_multiply(x, y, name=None):
                if False:
                    while True:
                        i = 10
                del x, y, name

    def testDispatchErrorNotCallable(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'Expected dispatch_target to be callable'):
            dispatch.dispatch_for_api(math_ops.abs, {0: MaskedTensor})('not_callable')

    def testRegisterDispatchableType(self):
        if False:
            while True:
                i = 10
        Car = collections.namedtuple('Car', ['size', 'speed'])
        dispatch.register_dispatchable_type(Car)

        @dispatch.dispatch_for_api(math_ops.add, {'x': Car, 'y': Car})
        def add_car(x, y, name=None):
            if False:
                return 10
            with ops.name_scope(name):
                return Car(x.size + y.size, x.speed + y.speed)
        try:
            x = Car(constant_op.constant(1), constant_op.constant(3))
            y = Car(constant_op.constant(10), constant_op.constant(20))
            z = math_ops.add(x, y)
            self.assertAllEqual(z.size, 11)
            self.assertAllEqual(z.speed, 23)
        finally:
            dispatch.unregister_dispatch_for(add_car)

    def testTypeCheckersAreCached(self):
        if False:
            print('Hello World!')
        checker1 = dispatch.make_type_checker(int)
        checker2 = dispatch.make_type_checker(int)
        self.assertIs(checker1, checker2)

    def testDispatchTargetWithNoNameArgument(self):
        if False:
            return 10

        @dispatch.dispatch_for_api(math_ops.add, {'x': MaskedTensor, 'y': MaskedTensor})
        def masked_add(x, y):
            if False:
                i = 10
                return i + 15
            return MaskedTensor(x.values + y.values, x.mask & y.mask)
        try:
            x = MaskedTensor([1, 2, 3, 4, 5], [1, 0, 1, 1, 1])
            y = MaskedTensor([1, 1, 1, 1, 1], [1, 1, 0, 1, 0])
            a = math_ops.add(x, y, name='MyAdd')
            if not context.executing_eagerly():
                self.assertRegex(a.values.name, '^MyAdd/add.*')
                self.assertRegex(a.mask.name, '^MyAdd/and.*')
            b = math_ops.add(x, y, 'B')
            if not context.executing_eagerly():
                self.assertRegex(b.values.name, '^B/add.*')
                self.assertRegex(b.mask.name, '^B/and.*')
            c = math_ops.add(x, y)
            if not context.executing_eagerly():
                self.assertRegex(c.values.name, '^add.*')
                self.assertRegex(c.mask.name, '^and.*')
        finally:
            dispatch.unregister_dispatch_for(masked_add)

    def testDispatchApiWithNoNameArg(self):
        if False:
            while True:
                i = 10
        signature = {'self': MaskedTensor, 'other': MaskedTensor}

        @dispatch.dispatch_for_api(math_ops.tensor_equals, signature)
        def masked_tensor_equals(self, other):
            if False:
                return 10
            del self, other
        dispatch.unregister_dispatch_for(masked_tensor_equals)
        with self.assertRaisesRegexp(ValueError, "Dispatch function's signature \\(self, other, name=None\\) does not match API's signature \\(self, other\\)\\."):

            @dispatch.dispatch_for_api(math_ops.tensor_equals, signature)
            def masked_tensor_equals_2(self, other, name=None):
                if False:
                    return 10
                del self, other, name
            del masked_tensor_equals_2

    def testDispatchWithIterableParams(self):
        if False:
            print('Hello World!')

        @dispatch.dispatch_for_api(math_ops.add_n, {'inputs': typing.List[MaskedTensor]})
        def masked_add_n(inputs):
            if False:
                while True:
                    i = 10
            masks = array_ops_stack.stack([x.mask for x in inputs])
            return MaskedTensor(math_ops.add_n([x.values for x in inputs]), math_ops.reduce_all(masks, axis=0))
        try:
            generator = (MaskedTensor([i], [True]) for i in range(5))
            y = math_ops.add_n(generator)
            self.assertAllEqual(y.values, [0 + 1 + 2 + 3 + 4])
            self.assertAllEqual(y.mask, [True])
        finally:
            dispatch.unregister_dispatch_for(masked_add_n)

    def testBadIterableParametersError(self):
        if False:
            while True:
                i = 10
        fn = lambda x: [t + 1 for t in x]
        with self.assertRaisesRegex(TypeError, 'iterable_parameters should be a list or tuple of string'):
            dispatch.add_dispatch_support(iterable_parameters='x')(fn)

    def testUnregisterDispatchTargetBadTargetError(self):
        if False:
            i = 10
            return i + 15
        fn = lambda x: x + 1
        with self.assertRaisesRegex(ValueError, 'Function .* was not registered'):
            dispatch.unregister_dispatch_for(fn)

    def testAddDuplicateApiDisptacherError(self):
        if False:
            while True:
                i = 10
        some_op = lambda x: x
        some_op = dispatch.add_type_based_api_dispatcher(some_op)
        with self.assertRaisesRegex(ValueError, '.* already has a type-based API dispatcher.'):
            some_op = dispatch.add_type_based_api_dispatcher(some_op)

    def testGetApisWithTypeBasedDispatch(self):
        if False:
            while True:
                i = 10
        dispatch_apis = dispatch.apis_with_type_based_dispatch()
        self.assertIn(math_ops.add, dispatch_apis)
        self.assertIn(array_ops.concat, dispatch_apis)

    def testTypeBasedDispatchTargetsFor(self):
        if False:
            i = 10
            return i + 15
        MaskedTensorList = typing.List[typing.Union[MaskedTensor, tensor_lib.Tensor]]
        try:

            @dispatch.dispatch_for_api(math_ops.add)
            def masked_add(x: MaskedTensor, y: MaskedTensor):
                if False:
                    i = 10
                    return i + 15
                del x, y

            @dispatch.dispatch_for_api(array_ops.concat)
            def masked_concat(values: MaskedTensorList, axis):
                if False:
                    while True:
                        i = 10
                del values, axis

            @dispatch.dispatch_for_api(math_ops.add)
            def silly_add(x: SillyTensor, y: SillyTensor):
                if False:
                    return 10
                del x, y

            @dispatch.dispatch_for_api(math_ops.abs)
            def silly_abs(x: SillyTensor):
                if False:
                    while True:
                        i = 10
                del x
            targets = dispatch.type_based_dispatch_signatures_for(MaskedTensor)
            expected = {math_ops.add: [{'x': MaskedTensor, 'y': MaskedTensor}], array_ops.concat: [{'values': MaskedTensorList}]}
            self.assertEqual(targets, expected)
        finally:
            dispatch.unregister_dispatch_for(masked_add)
            dispatch.unregister_dispatch_for(masked_concat)
            dispatch.unregister_dispatch_for(silly_add)
            dispatch.unregister_dispatch_for(silly_abs)

    def testDispatchForUnaryElementwiseAPIs(self):
        if False:
            return 10

        @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
        def unary_elementwise_api_handler(api_func, x):
            if False:
                for i in range(10):
                    print('nop')
            return MaskedTensor(api_func(x.values), x.mask)
        try:
            x = MaskedTensor([1, -2, -3], [True, True, False])
            abs_x = math_ops.abs(x)
            sign_x = math_ops.sign(x=x)
            neg_x = math_ops.negative(x, 'neg_x')
            invert_x = bitwise_ops.invert(x, name='invert_x')
            ones_like_x = array_ops.ones_like(x, name='ones_like_x')
            ones_like_x_float = array_ops.ones_like(x, dtypes.float32, name='ones_like_x_float')
            self.assertAllEqual(abs_x.values, [1, 2, 3])
            self.assertAllEqual(sign_x.values, [1, -1, -1])
            self.assertAllEqual(neg_x.values, [-1, 2, 3])
            self.assertAllEqual(invert_x.values, [-2, 1, 2])
            self.assertAllEqual(ones_like_x.values, [1, 1, 1])
            self.assertAllEqual(ones_like_x_float.values, [1.0, 1.0, 1.0])
            for result in [abs_x, sign_x, neg_x, invert_x, ones_like_x, ones_like_x_float]:
                self.assertAllEqual(result.mask, [True, True, False])
            if not context.executing_eagerly():
                self.assertRegex(neg_x.values.name, '^neg_x/Neg:.*')
                self.assertRegex(invert_x.values.name, '^invert_x/.*')
                self.assertRegex(ones_like_x.values.name, '^ones_like_x/.*')
                self.assertRegex(ones_like_x_float.values.name, '^ones_like_x_float/.*')
        finally:
            dispatch.unregister_dispatch_for(unary_elementwise_api_handler)

    def testDispatchForBinaryElementwiseAPIs(self):
        if False:
            return 10

        @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
        def binary_elementwise_api_handler(api_func, x, y):
            if False:
                print('Hello World!')
            return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
        try:
            x = MaskedTensor([1, -2, -3], [True, True, False])
            y = MaskedTensor([10, 20, 30], [True, False, True])
            x_times_y = math_ops.multiply(x, y)
            x_plus_y = math_ops.add(x, y=y)
            x_minus_y = math_ops.subtract(x=x, y=y)
            min_x_y = math_ops.minimum(x, y, 'min_x_y')
            y_times_x = math_ops.multiply(y, x, name='y_times_x')
            y_plus_x = math_ops.add(y, y=x, name='y_plus_x')
            y_minus_x = math_ops.subtract(x=y, y=x, name='y_minus_x')
            self.assertAllEqual(x_times_y.values, [10, -40, -90])
            self.assertAllEqual(x_plus_y.values, [11, 18, 27])
            self.assertAllEqual(x_minus_y.values, [-9, -22, -33])
            self.assertAllEqual(min_x_y.values, [1, -2, -3])
            self.assertAllEqual(y_times_x.values, [10, -40, -90])
            self.assertAllEqual(y_plus_x.values, [11, 18, 27])
            self.assertAllEqual(y_minus_x.values, [9, 22, 33])
            for result in [x_times_y, x_plus_y, x_minus_y, min_x_y, y_times_x, y_plus_x, y_minus_x]:
                self.assertAllEqual(result.mask, [True, False, False])
            if not context.executing_eagerly():
                self.assertRegex(min_x_y.values.name, '^min_x_y/Minimum:.*')
                self.assertRegex(min_x_y.mask.name, '^min_x_y/and:.*')
                self.assertRegex(y_times_x.values.name, '^y_times_x/.*')
                self.assertRegex(y_plus_x.values.name, '^y_plus_x/.*')
                self.assertRegex(y_minus_x.values.name, '^y_minus_x/.*')
        finally:
            dispatch.unregister_dispatch_for(binary_elementwise_api_handler)

    def testDuplicateDispatchForUnaryElementwiseAPIsError(self):
        if False:
            return 10

        @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
        def handler(api_func, x):
            if False:
                print('Hello World!')
            return MaskedTensor(api_func(x.values), x.mask)
        try:
            with self.assertRaisesRegex(ValueError, 'A unary elementwise dispatch handler \\(.*\\) has already been registered for .*'):

                @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
                def another_handler(api_func, x):
                    if False:
                        i = 10
                        return i + 15
                    return MaskedTensor(api_func(x.values), ~x.mask)
                del another_handler
        finally:
            dispatch.unregister_dispatch_for(handler)

    def testDuplicateDispatchForBinaryElementwiseAPIsError(self):
        if False:
            i = 10
            return i + 15

        @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
        def handler(api_func, x, y):
            if False:
                i = 10
                return i + 15
            return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
        try:
            with self.assertRaisesRegex(ValueError, 'A binary elementwise dispatch handler \\(.*\\) has already been registered for .*'):

                @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
                def another_handler(api_func, x, y):
                    if False:
                        i = 10
                        return i + 15
                    return MaskedTensor(api_func(x.values, y.values), x.mask)
                del another_handler
        finally:
            dispatch.unregister_dispatch_for(handler)

    def testRegisterUnaryElementwiseApiAfterHandler(self):
        if False:
            print('Hello World!')

        @dispatch.dispatch_for_unary_elementwise_apis(MaskedTensor)
        def handler(api_func, x):
            if False:
                while True:
                    i = 10
            return MaskedTensor(api_func(x.values), x.mask)
        try:

            @dispatch.register_unary_elementwise_api
            @dispatch.add_dispatch_support
            def some_op(x):
                if False:
                    i = 10
                    return i + 15
                return x * 2
            x = MaskedTensor([1, 2, 3], [True, False, True])
            y = some_op(x)
            self.assertAllEqual(y.values, [2, 4, 6])
            self.assertAllEqual(y.mask, [True, False, True])
        finally:
            dispatch.unregister_dispatch_for(handler)

    def testRegisterBinaryElementwiseApiAfterHandler(self):
        if False:
            i = 10
            return i + 15

        @dispatch.dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
        def handler(api_func, x, y):
            if False:
                while True:
                    i = 10
            return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
        try:

            @dispatch.register_binary_elementwise_api
            @dispatch.add_dispatch_support
            def some_op(x, y):
                if False:
                    i = 10
                    return i + 15
                return x * 2 + y
            x = MaskedTensor([1, 2, 3], [True, False, True])
            y = MaskedTensor([10, 20, 30], [True, True, False])
            z = some_op(x, y)
            self.assertAllEqual(z.values, [12, 24, 36])
            self.assertAllEqual(z.mask, [True, False, False])
        finally:
            dispatch.unregister_dispatch_for(handler)

    def testElementwiseApiLists(self):
        if False:
            i = 10
            return i + 15
        self.assertIn(math_ops.abs, dispatch.unary_elementwise_apis())
        self.assertIn(math_ops.cos, dispatch.unary_elementwise_apis())
        self.assertIn(math_ops.add, dispatch.binary_elementwise_apis())
        self.assertIn(math_ops.multiply, dispatch.binary_elementwise_apis())

    def testUpdateDocstringsWithAPILists(self):
        if False:
            i = 10
            return i + 15
        dispatch.update_docstrings_with_api_lists()
        self.assertRegex(dispatch.dispatch_for_api.__doc__, '(?s)  The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:\\n\\n.*  \\* `tf\\.concat\\(values, axis, name\\)`\\n.*  \\* `tf\\.math\\.add\\(x, y, name\\)`\\n.*')
        self.assertRegex(dispatch.dispatch_for_unary_elementwise_apis.__doc__, '(?s)  The unary elementwise APIs are:\\n\\n.*  \\* `tf\\.math\\.abs\\(x, name\\)`\\n.*  \\* `tf\\.math\\.cos\\(x, name\\)`\\n.*')
        self.assertRegex(dispatch.dispatch_for_binary_elementwise_apis.__doc__, '(?s)  The binary elementwise APIs are:\\n\\n.*  \\* `tf\\.math\\.add\\(x, y, name\\)`\\n.*  \\* `tf\\.math\\.multiply\\(x, y, name\\)`\\n.*')
if __name__ == '__main__':
    googletest.main()
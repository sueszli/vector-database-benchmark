from absl.testing import parameterized
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class ConcreteFunctionTest(test.TestCase, parameterized.TestCase):

    def concrete_function_with_attrs(self, attrs):
        if False:
            print('Hello World!')
        func_graph = func_graph_module.FuncGraph('f')
        return cf.ConcreteFunction.from_func_graph(func_graph, None, attrs=attrs)

    @parameterized.parameters(({'api_implements': True}, attr_value_pb2.AttrValue(b=True)), ({'api_implements': 1}, attr_value_pb2.AttrValue(i=1)), ({'api_implements': 1.0}, attr_value_pb2.AttrValue(f=1.0)), ({'api_implements': 'test'}, attr_value_pb2.AttrValue(s=compat.as_bytes('test'))))
    def test_parses_func_attr_scalar_values(self, attrs, expected):
        if False:
            return 10
        self.assertEqual(self.concrete_function_with_attrs(attrs=attrs).function_def.attr['api_implements'], expected)

    def test_parses_func_attr_list_values(self):
        if False:
            print('Hello World!')
        self.assertProtoEquals("\n        list {\n            s: 'test'\n            b: True\n            i: 1\n            f: 1.0\n        }\n        ", self.concrete_function_with_attrs(attrs={'api_implements': ['test', True, 1, 1.0]}).function_def.attr['api_implements'])

    def test_raises_value_error_for_invalid_attr(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Attribute api_implements must be'):
            self.concrete_function_with_attrs(attrs={'api_implements': None})

    def test_generate_from_atomic(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def add_dicts(dict_a, dict_b):
            if False:
                print('Hello World!')
            result = {}
            for key in dict_a.keys():
                result[key] = dict_a[key] + dict_b[key]
            return result
        dict_a = {'tensor': constant_op.constant(1), 'variable': variables.Variable(2), 'ragged_tensor': ragged_tensor.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8]), 'python_int': 4}
        dict_b = {'tensor': constant_op.constant(2), 'variable': variables.Variable(5), 'ragged_tensor': ragged_tensor.RaggedTensor.from_row_splits(values=[4, 2, 4, 1, 6, 9, 3, 6], row_splits=[0, 4, 4, 7, 8, 8]), 'python_int': 5}
        original_concrete_fn = add_dicts.get_concrete_function(dict_a, dict_b)
        atomic_fn = original_concrete_fn._inference_function
        del add_dicts
        del original_concrete_fn
        concrete_fn = cf.ConcreteFunction(atomic_fn)
        result = concrete_fn(dict_a, dict_b)
        self.assertEqual(result['tensor'].numpy(), 3)
        self.assertEqual(result['variable'].numpy(), 7)
        self.assertEqual(result['ragged_tensor'].flat_values.numpy().tolist(), [7, 3, 8, 2, 11, 18, 5, 12])
        self.assertEqual(result['python_int'].numpy(), 9)

    def test_generate_from_def(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def add_dicts(dict_a, dict_b):
            if False:
                for i in range(10):
                    print('nop')
            result = {}
            for key in dict_a.keys():
                result[key] = dict_a[key] + dict_b[key]
            return result
        dict_a = {'tensor': constant_op.constant(1), 'variable': variables.Variable(2), 'python_int': 4}
        dict_b = {'tensor': constant_op.constant(2), 'variable': variables.Variable(5), 'python_int': 5}
        original_concrete_fn = add_dicts.get_concrete_function(dict_a, dict_b)
        function_def = original_concrete_fn.function_def
        function_type = original_concrete_fn.function_type
        del add_dicts
        del original_concrete_fn
        atomic_fn = atomic_function.from_function_def(function_def, function_type)
        concrete_fn = cf.ConcreteFunction(atomic_fn)
        result = concrete_fn(dict_a, dict_b)
        self.assertEqual(result['tensor'].numpy(), 3)
        self.assertEqual(result['variable'].numpy(), 7)
        self.assertEqual(result['python_int'].numpy(), 9)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()
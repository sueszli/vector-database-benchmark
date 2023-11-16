import tree
from keras import testing
from keras.backend import KerasTensor
from keras.ops.symbolic_arguments import SymbolicArguments

class SymbolicArgumentsTest(testing.TestCase):

    def test_args(self):
        if False:
            while True:
                i = 10
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        args = SymbolicArguments((a, b), {})
        self.assertEqual(args.keras_tensors, [a, b])
        self.assertEqual(args._flat_arguments, [a, b])
        self.assertEqual(args._single_positional_tensor, None)

    def test_args_single_arg(self):
        if False:
            print('Hello World!')
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        args = SymbolicArguments(a)
        self.assertEqual(args.keras_tensors, [a])
        self.assertEqual(args._flat_arguments, [a])
        self.assertEqual(len(args.kwargs), 0)
        self.assertEqual(isinstance(args.args[0], KerasTensor), True)
        self.assertEqual(args._single_positional_tensor, a)

    def test_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        c = KerasTensor(shape=shape)
        args = SymbolicArguments((a, b), {1: c})
        self.assertEqual(args.keras_tensors, [a, b, c])
        self.assertEqual(args._flat_arguments, [a, b, c])
        self.assertEqual(args._single_positional_tensor, None)

    def test_conversion_fn(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        c = KerasTensor(shape=shape)
        sym_args = SymbolicArguments((a, b), {1: c})
        (value, _) = sym_args.convert(lambda x: x ** 2)
        args1 = value[0][0]
        self.assertIsInstance(args1, KerasTensor)
        mapped_value = tree.map_structure(lambda x: x ** 2, a)
        self.assertEqual(mapped_value.shape, args1.shape)
        self.assertEqual(mapped_value.dtype, args1.dtype)

    def test_fill_in_single_arg(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        tensor_dict = {id(a): 3}
        sym_args = SymbolicArguments(a)
        (result, _) = sym_args.fill_in(tensor_dict)
        self.assertEqual(result, (3,))

    def test_fill_in_multiple_arg(self):
        if False:
            print('Hello World!')
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        tensor_dict = {id(b): 2}
        sym_args = SymbolicArguments((a, b))
        (result, _) = sym_args.fill_in(tensor_dict)
        self.assertEqual(result, ((a, 2),))

    def test_fill_in(self):
        if False:
            for i in range(10):
                print('nop')
        shape1 = (2, 3, 4)
        shape2 = (3, 2, 4)
        a = KerasTensor(shape=shape1)
        b = KerasTensor(shape=shape2)
        c = KerasTensor(shape=shape2)
        dictionary = {id(a): 3, id(c): 2}
        sym_args = SymbolicArguments((a, b), {1: c})
        (values, _) = sym_args.fill_in(dictionary)
        self.assertEqual(values, ((3, b), {1: 2}))
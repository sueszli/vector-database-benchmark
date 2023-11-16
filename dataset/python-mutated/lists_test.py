"""Tests for lists module."""
from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test

class ListTest(converter_testing.TestCase):

    def test_empty_list(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return []
        tr = self.transform(f, lists)
        tl = tr()
        self.assertIsInstance(tl, tensor.Tensor)
        self.assertEqual(tl.dtype, dtypes.variant)

    def test_initialized_list(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return [1, 2, 3]
        tr = self.transform(f, lists)
        self.assertAllEqual(tr(), [1, 2, 3])

    def test_list_append(self):
        if False:
            return 10

        def f():
            if False:
                print('Hello World!')
            l = special_functions.tensor_list([1])
            l.append(2)
            l.append(3)
            return l
        tr = self.transform(f, lists)
        tl = tr()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(self.evaluate(r), [1, 2, 3])

    def test_list_pop(self):
        if False:
            return 10

        def f():
            if False:
                while True:
                    i = 10
            l = special_functions.tensor_list([1, 2, 3])
            directives.set_element_type(l, dtype=dtypes.int32, shape=())
            s = l.pop()
            return (s, l)
        tr = self.transform(f, (directives_converter, lists))
        (ts, tl) = tr()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(self.evaluate(r), [1, 2])
        self.assertAllEqual(self.evaluate(ts), 3)

    def test_double_list_pop(self):
        if False:
            print('Hello World!')

        def f(l):
            if False:
                i = 10
                return i + 15
            s = l.pop().pop()
            return s
        tr = self.transform(f, lists)
        test_input = [1, 2, [1, 2, 3]]
        self.assertAllEqual(tr(test_input), 3)

    def test_list_stack(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                return 10
            l = [1, 2, 3]
            return array_ops_stack.stack(l)
        tr = self.transform(f, lists)
        self.assertAllEqual(self.evaluate(tr()), [1, 2, 3])
if __name__ == '__main__':
    test.main()
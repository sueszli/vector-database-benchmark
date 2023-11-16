import base64
import marshal
from keras import testing
from keras.utils import python_utils

class PythonUtilsTest(testing.TestCase):

    def test_func_dump_and_load(self):
        if False:
            while True:
                i = 10

        def my_function(x, y=1, **kwargs):
            if False:
                i = 10
                return i + 15
            return x + y
        serialized = python_utils.func_dump(my_function)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(2, y=3), 5)

    def test_removesuffix(self):
        if False:
            while True:
                i = 10
        x = 'model.keras'
        self.assertEqual(python_utils.removesuffix(x, '.keras'), 'model')
        self.assertEqual(python_utils.removesuffix(x, 'model'), x)

    def test_removeprefix(self):
        if False:
            while True:
                i = 10
        x = 'model.keras'
        self.assertEqual(python_utils.removeprefix(x, 'model'), '.keras')
        self.assertEqual(python_utils.removeprefix(x, '.keras'), x)

    def test_func_load_defaults_as_tuple(self):
        if False:
            while True:
                i = 10

        def dummy_function(x=(1, 2, 3)):
            if False:
                while True:
                    i = 10
            pass
        serialized = python_utils.func_dump(dummy_function)
        deserialized = python_utils.func_load(serialized)
        self.assertIsInstance(deserialized.__defaults__[0], tuple)
        self.assertEqual(deserialized.__defaults__[0], (1, 2, 3))

    def test_remove_long_seq_standard_case(self):
        if False:
            for i in range(10):
                print('nop')
        sequences = [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]
        labels = [1, 2, 3, 4]
        (new_sequences, new_labels) = python_utils.remove_long_seq(3, sequences, labels)
        self.assertEqual(new_sequences, [[1], [2, 2]])
        self.assertEqual(new_labels, [1, 2])

    def test_func_load_with_closure(self):
        if False:
            return 10

        def outer_fn(x):
            if False:
                while True:
                    i = 10

            def inner_fn(y):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y
            return inner_fn
        func_with_closure = outer_fn(10)
        serialized = python_utils.func_dump(func_with_closure)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(5), 15)

    def test_func_load_closure_conversion(self):
        if False:
            return 10

        def my_function_with_closure(x):
            if False:
                i = 10
                return i + 15
            return x + y
        y = 5
        serialized = python_utils.func_dump(my_function_with_closure)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(5), 10)

    def test_ensure_value_to_cell(self):
        if False:
            return 10
        value_to_test = 'test_value'

        def dummy_fn():
            if False:
                while True:
                    i = 10
            value_to_test
        cell_value = dummy_fn.__closure__[0].cell_contents
        self.assertEqual(value_to_test, cell_value)

    def test_closure_processing(self):
        if False:
            while True:
                i = 10

        def simple_function(x):
            if False:
                print('Hello World!')
            return x + 10
        serialized = python_utils.func_dump(simple_function)
        deserialized = python_utils.func_load(serialized)
        self.assertEqual(deserialized(5), 15)

    def test_func_load_valid_encoded_code(self):
        if False:
            while True:
                i = 10

        def another_simple_function(x):
            if False:
                while True:
                    i = 10
            return x * 2
        raw_data = marshal.dumps(another_simple_function.__code__)
        valid_encoded_code = base64.b64encode(raw_data).decode('utf-8')
        try:
            python_utils.func_load(valid_encoded_code)
        except (UnicodeEncodeError, ValueError):
            self.fail('Expected no error for valid code, but got an error.')

    def test_func_load_bad_encoded_code(self):
        if False:
            i = 10
            return i + 15
        bad_encoded_code = "This isn't valid base64!"
        with self.assertRaises(AttributeError):
            python_utils.func_load(bad_encoded_code)
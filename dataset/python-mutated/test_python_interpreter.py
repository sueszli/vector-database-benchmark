import unittest
from transformers.testing_utils import CaptureStdout
from transformers.tools.python_interpreter import evaluate

def add_two(x):
    if False:
        return 10
    return x + 2

class PythonInterpreterTester(unittest.TestCase):

    def test_evaluate_assign(self):
        if False:
            print('Hello World!')
        code = 'x = 3'
        state = {}
        result = evaluate(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {'x': 3})
        code = 'x = y'
        state = {'y': 5}
        result = evaluate(code, {}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 5, 'y': 5})

    def test_evaluate_call(self):
        if False:
            print('Hello World!')
        code = 'y = add_two(x)'
        state = {'x': 3}
        result = evaluate(code, {'add_two': add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 3, 'y': 5})
        with CaptureStdout() as out:
            result = evaluate(code, {}, state=state)
        assert result is None
        assert 'tried to execute add_two' in out.out

    def test_evaluate_constant(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'x = 3'
        state = {}
        result = evaluate(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {'x': 3})

    def test_evaluate_dict(self):
        if False:
            return 10
        code = "test_dict = {'x': x, 'y': add_two(x)}"
        state = {'x': 3}
        result = evaluate(code, {'add_two': add_two}, state=state)
        self.assertDictEqual(result, {'x': 3, 'y': 5})
        self.assertDictEqual(state, {'x': 3, 'test_dict': {'x': 3, 'y': 5}})

    def test_evaluate_expression(self):
        if False:
            print('Hello World!')
        code = 'x = 3\ny = 5'
        state = {}
        result = evaluate(code, {}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 3, 'y': 5})

    def test_evaluate_f_string(self):
        if False:
            while True:
                i = 10
        code = "text = f'This is x: {x}.'"
        state = {'x': 3}
        result = evaluate(code, {}, state=state)
        assert result == 'This is x: 3.'
        self.assertDictEqual(state, {'x': 3, 'text': 'This is x: 3.'})

    def test_evaluate_if(self):
        if False:
            print('Hello World!')
        code = 'if x <= 3:\n    y = 2\nelse:\n    y = 5'
        state = {'x': 3}
        result = evaluate(code, {}, state=state)
        assert result == 2
        self.assertDictEqual(state, {'x': 3, 'y': 2})
        state = {'x': 8}
        result = evaluate(code, {}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 8, 'y': 5})

    def test_evaluate_list(self):
        if False:
            return 10
        code = 'test_list = [x, add_two(x)]'
        state = {'x': 3}
        result = evaluate(code, {'add_two': add_two}, state=state)
        self.assertListEqual(result, [3, 5])
        self.assertDictEqual(state, {'x': 3, 'test_list': [3, 5]})

    def test_evaluate_name(self):
        if False:
            return 10
        code = 'y = x'
        state = {'x': 3}
        result = evaluate(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {'x': 3, 'y': 3})

    def test_evaluate_subscript(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'test_list = [x, add_two(x)]\ntest_list[1]'
        state = {'x': 3}
        result = evaluate(code, {'add_two': add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 3, 'test_list': [3, 5]})
        code = "test_dict = {'x': x, 'y': add_two(x)}\ntest_dict['y']"
        state = {'x': 3}
        result = evaluate(code, {'add_two': add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {'x': 3, 'test_dict': {'x': 3, 'y': 5}})

    def test_evaluate_for(self):
        if False:
            i = 10
            return i + 15
        code = 'x = 0\nfor i in range(3):\n    x = i'
        state = {}
        result = evaluate(code, {'range': range}, state=state)
        assert result == 2
        self.assertDictEqual(state, {'x': 2, 'i': 2})
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
try:
    from . import utils
except ImportError:
    import utils

class Pair:

    def __init__(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y

def Foo():
    if False:
        print('Hello World!')
    return Pair(1, 1)
g_counter = 1
g_list = [0, 1, 2]
g_dict = {'a': 0, 'b': 1}
g_object = Foo()
g_tensor = torch.zeros(10)
_name: int = 0

def fresh_name() -> str:
    if False:
        print('Hello World!')
    'create a new unique name for a variable: v0, v1, v2'
    global _name
    r = f'v{_name}'
    _name += 1
    return r

def reset_name():
    if False:
        return 10
    global _name
    _name = 0

class TestGlobals(torch._dynamo.test_case.TestCase):

    def test_store_global_1(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                print('Hello World!')
            global g_counter
            val = x + g_counter
            g_counter += 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_2(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                i = 10
                return i + 15
            global g_counter
            val = x + g_counter
            g_counter += 1
            g_counter += 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        'Wrap the second call with torch._dynamo as well'
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res2 = opt_fn(x)
        self.assertTrue(same(res2 - res1, 2 * torch.ones(10)))

    def test_store_global_new(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                return 10
            global g_counter_new
            g_counter_new = x + 1
            return x + g_counter_new
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        self.assertTrue(same(res1, x + x + 1))

    def test_store_global_list(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                print('Hello World!')
            global g_list
            val = x + g_list[1]
            '\n            Strictly speaking, we are not testing STORE_GLOBAL\n            here, since STORE_SUBSCR is actually used to store.\n            '
            g_list[1] += 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_list_2(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            global g_list
            val = x + g_list[1]
            g_list = [x + 1 for x in g_list]
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            global g_dict
            val = x + g_dict['b']
            '\n            Strictly speaking, we are not testing STORE_GLOBAL\n            here, since STORE_SUBSCR is actually used to store.\n            '
            g_dict['b'] += 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict_2(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                print('Hello World!')
            global g_dict
            g_dict = {key: value + 1 for (key, value) in g_dict.items()}
            val = x + g_dict['b']
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_object(self):
        if False:
            return 10

        def fn(x):
            if False:
                return 10
            global g_object
            val = x + g_object.y
            g_object.y += 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_cross_file(self):
        if False:
            return 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            val = x + utils.g_tensor_export
            utils.g_tensor_export = utils.g_tensor_export + 1
            return val
        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_inline_1(self):
        if False:
            while True:
                i = 10

        class Variable:

            def __init__(self, value: torch.Tensor, name: str=None):
                if False:
                    while True:
                        i = 10
                self.value = value
                self.name = name or fresh_name()

        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = Variable(a)
            b = Variable(b)
            return (a.value + b.value, a.name + b.name)
        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        (v0, s0) = opt_fn(a, b)
        self.assertEqual(s0, 'v0v1')
        reset_name()

    def test_store_global_inline_2(self):
        if False:
            for i in range(10):
                print('nop')

        class Variable:

            def __init__(self, value: torch.Tensor, name: str=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value
                self.name = name or fresh_name()

            @staticmethod
            def constant(value: torch.Tensor, name: str=None):
                if False:
                    while True:
                        i = 10
                return Variable(value, name)

        def fn(a, b):
            if False:
                return 10
            a = Variable.constant(a)
            b = Variable.constant(b)
            return (a.value + b.value, a.name + b.name)
        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        (v0, s0) = opt_fn(a, b)
        self.assertEqual(s0, 'v0v1')
        reset_name()
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()
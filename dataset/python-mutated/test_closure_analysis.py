import inspect
import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
from numpy import append
import paddle
from paddle.jit.dy2static.utils import FunctionNameLivenessAnalysis
from paddle.utils import gast
global_a = []

class JudgeVisitor(gast.NodeVisitor):

    def __init__(self, ans, mod):
        if False:
            while True:
                i = 10
        self.ans = ans
        self.mod = mod

    def visit_FunctionDef(self, node):
        if False:
            while True:
                i = 10
        scope = node.pd_scope
        expected = self.ans.get(node.name, set())
        exp_mod = self.mod.get(node.name, set())
        assert scope.existed_vars() == expected, 'Not Equals.'
        assert scope.modified_vars() == exp_mod, 'Not Equals in function:{} . expect {} , but get {}'.format(node.name, exp_mod, scope.modified_vars())
        self.generic_visit(node)

class JudgePushPopVisitor(gast.NodeVisitor):

    def __init__(self, push_pop_vars):
        if False:
            while True:
                i = 10
        self.pp_var = push_pop_vars

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        scope = node.pd_scope
        expected = self.pp_var.get(node.name, set())
        assert scope.push_pop_vars == expected, 'Not Equals in function:{} . expect {} , but get {}'.format(node.name, expected, scope.push_pop_vars)
        self.generic_visit(node)

def test_normal_0(x):
    if False:
        for i in range(10):
            print('nop')

    def func():
        if False:
            while True:
                i = 10
        if True:
            i = 1
    func()
    return i

def test_normal_argument(x):
    if False:
        for i in range(10):
            print('nop')
    x = 1

    def func():
        if False:
            for i in range(10):
                print('nop')
        if True:
            print(x)
            i = 1
    func()
    return x

def test_global(x):
    if False:
        return 10
    global t
    t = 10

    def func():
        if False:
            while True:
                i = 10
        if True:
            print(x)
            i = 1
    func()
    return x

def test_nonlocal(x, *args, **kargs):
    if False:
        for i in range(10):
            print('nop')
    i = 10

    def func(*args, **kargs):
        if False:
            i = 10
            return i + 15
        nonlocal i
        k = 10
        if True:
            print(x)
            i = 1
    func(*args, **kargs)
    return x

def test_push_pop_1(x, *args, **kargs):
    if False:
        while True:
            i = 10
    'push_pop_vars in main_function is : `l`, `k`'
    l = []
    k = []
    for i in range(10):
        l.append(i)
        k.pop(i)
    return l

def test_push_pop_2(x, *args, **kargs):
    if False:
        print('Hello World!')
    'push_pop_vars in main_function is : `k`'
    l = []
    k = []

    def func():
        if False:
            i = 10
            return i + 15
        l.append(0)
    for i in range(10):
        k.append(i)
    return (l, k)

def test_push_pop_3(x, *args, **kargs):
    if False:
        print('Hello World!')
    'push_pop_vars in main_function is : `k`\n    NOTE: One may expect `k` and `l` because l\n          is nonlocal. Name bind analysis is\n          not implemented yet.\n    '
    l = []
    k = []

    def func():
        if False:
            print('Hello World!')
        nonlocal l
        l.append(0)
    for i in range(10):
        k.append(i)
    return (l, k)

def test_push_pop_4(x, *args, **kargs):
    if False:
        for i in range(10):
            print('nop')
    'push_pop_vars in main_function is : `k`'
    l = []
    k = []
    for i in range(10):
        for j in range(10):
            if True:
                l.append(j)
            else:
                k.pop()
    return (l, k)

class TestClosureAnalysis(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.judge_type = 'var and w_vars'
        self.init_dygraph_func()

    def init_dygraph_func(self):
        if False:
            while True:
                i = 10
        self.all_dygraph_funcs = [test_nonlocal, test_global, test_normal_0, test_normal_argument]
        self.answer = [{'func': set('k'), 'test_nonlocal': set('i')}, {'func': set({'i'})}, {'func': set('i')}, {'func': set('i')}]
        self.modified_var = [{'func': set('ki'), 'test_nonlocal': set('i')}, {'func': set({'i'}), 'test_global': set({'t'})}, {'func': set('i')}, {'func': set('i'), 'test_normal_argument': set('x')}]

    def test_main(self):
        if False:
            i = 10
            return i + 15
        if self.judge_type == 'push_pop_vars':
            for (push_pop_vars, func) in zip(self.push_pop_vars, self.all_dygraph_funcs):
                test_func = inspect.getsource(func)
                gast_root = gast.parse(test_func)
                name_visitor = FunctionNameLivenessAnalysis(gast_root)
                JudgePushPopVisitor(push_pop_vars).visit(gast_root)
        else:
            for (mod, ans, func) in zip(self.modified_var, self.answer, self.all_dygraph_funcs):
                test_func = inspect.getsource(func)
                gast_root = gast.parse(test_func)
                name_visitor = FunctionNameLivenessAnalysis(gast_root)
                JudgeVisitor(ans, mod).visit(gast_root)

def TestClosureAnalysis_Attribute_func():
    if False:
        while True:
            i = 10
    i = 0
    self.current.function = 12

class TestClosureAnalysis_Attribute(TestClosureAnalysis):

    def init_dygraph_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.all_dygraph_funcs = [TestClosureAnalysis_Attribute_func]
        self.answer = [{'TestClosureAnalysis_Attribute_func': set({'i'})}]
        self.modified_var = [{'TestClosureAnalysis_Attribute_func': set({'i', 'self.current.function'})}]

class TestClosureAnalysis_PushPop(TestClosureAnalysis):

    def init_dygraph_func(self):
        if False:
            i = 10
            return i + 15
        self.judge_type = 'push_pop_vars'
        self.all_dygraph_funcs = [test_push_pop_1, test_push_pop_2, test_push_pop_3, test_push_pop_4]
        self.push_pop_vars = [{'test_push_pop_1': set({'l', 'k'})}, {'test_push_pop_2': set({'k'}), 'func': set('l')}, {'test_push_pop_3': set({'k'}), 'func': set('l')}, {'test_push_pop_4': set({'k', 'l'})}]

class TestPushPopTrans(Dy2StTestBase):

    @test_legacy_and_pir
    def test(self):
        if False:
            return 10

        def vlist_of_dict(x):
            if False:
                i = 10
                return i + 15
            ma = {'a': []}
            for i in range(3):
                ma['a'].append(1)
            return ma
        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict)(x))

    @test_legacy_and_pir
    def test2(self):
        if False:
            return 10
        import numpy as np

        def vlist_of_dict(x):
            if False:
                i = 10
                return i + 15
            a = np.array([1, 2, 3])
            for i in range(3):
                np.append(a, 4)
            return a
        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict)(x))

    @test_legacy_and_pir
    def test3(self):
        if False:
            i = 10
            return i + 15
        import numpy as np

        def vlist_of_dict(x):
            if False:
                for i in range(10):
                    print('nop')
            a = np.array([1, 2, 3])
            if True:
                pass
            return a
        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict)(x))

    @test_legacy_and_pir
    def test4(self):
        if False:
            print('Hello World!')
        import numpy as np

        def vlist_of_dict(x):
            if False:
                while True:
                    i = 10
            a = np.array([1, 2, 3])
            for i in range(3):
                append(a, 4)
            return a
        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict)(x))

    @test_legacy_and_pir
    def test5(self):
        if False:
            i = 10
            return i + 15
        import numpy as np

        def vlist_of_dict(x):
            if False:
                print('Hello World!')
            a = np.array([1, 2, 3])
            for i in range(3):
                global_a.append(4)
            return a
        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict)(x))
if __name__ == '__main__':
    unittest.main()
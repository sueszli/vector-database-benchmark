import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_sot_only
import paddle

def func_loop_write_dict(x):
    if False:
        for i in range(10):
            print('nop')
    res = {'a': 1}
    t = paddle.shape(x)[0]
    for i in range(t):
        res['a'] = i
    return res

def func_loop_write_list(x):
    if False:
        for i in range(10):
            print('nop')
    res = [1]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0] = i
    return res

def func_loop_write_nest_dict_list(x):
    if False:
        while True:
            i = 10
    res = {'a': [1]}
    t = paddle.shape(x)[0]
    for i in range(t):
        res['a'][0] = i
    return res

def func_loop_write_nest_list_dict(x):
    if False:
        while True:
            i = 10
    res = [{'a': 1}]
    t = paddle.shape(x)[0]
    for i in range(t):
        res[0]['a'] = i
    return res

def func_ifelse_write_dict(x):
    if False:
        print('Hello World!')
    res = {'a': 1}
    t = paddle.shape(x)[0]
    if t > 2:
        res['a'] = 2
    else:
        res['a'] = 3
    return res

def func_ifelse_write_list(x):
    if False:
        i = 10
        return i + 15
    res = [1]
    t = paddle.shape(x)[0]
    if t > 2:
        res[0] = 2
    else:
        res[0] = 3
    return res

def func_ifelse_write_nest_dict_list(x):
    if False:
        print('Hello World!')
    res = {'a': [1]}
    t = paddle.shape(x)[0]
    if t > 2:
        res['a'][0] = 2
    else:
        res['a'][0] = 3
    return res

def func_ifelse_write_nest_list_dict(x):
    if False:
        for i in range(10):
            print('nop')
    res = [{'a': 1}]
    t = paddle.shape(x)[0]
    if t > 2:
        res[0]['a'] = 2
    else:
        res[0]['a'] = 3
    return res

class TestWriteContainer(Dy2StTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_func()
        self.set_getitem_path()

    def set_func(self):
        if False:
            while True:
                i = 10
        self.func = func_loop_write_dict

    def set_getitem_path(self):
        if False:
            print('Hello World!')
        self.getitem_path = ('a',)

    def get_raw_value(self, container, getitem_path):
        if False:
            return 10
        out = container
        for path in getitem_path:
            out = out[path]
        return out

    @test_sot_only
    def test_write_container_sot(self):
        if False:
            return 10
        func_static = paddle.jit.to_static(self.func)
        input = paddle.to_tensor([1, 2, 3])
        out_static = self.get_raw_value(func_static(input), self.getitem_path)
        out_dygraph = self.get_raw_value(self.func(input), self.getitem_path)
        self.assertEqual(out_static, out_dygraph)

    @test_ast_only
    def test_write_container(self):
        if False:
            return 10
        func_static = paddle.jit.to_static(self.func)
        input = paddle.to_tensor([1, 2, 3])
        out_static = self.get_raw_value(func_static(input), self.getitem_path).item()
        out_dygraph = self.get_raw_value(self.func(input), self.getitem_path)
        self.assertEqual(out_static, out_dygraph)

class TestLoopWriteContainerList(TestWriteContainer):

    def set_func(self):
        if False:
            print('Hello World!')
        self.func = func_loop_write_list

    def set_getitem_path(self):
        if False:
            print('Hello World!')
        self.getitem_path = (0,)

class TestLoopWriteContainerNestDictList(TestWriteContainer):

    def set_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.func = func_loop_write_nest_dict_list

    def set_getitem_path(self):
        if False:
            i = 10
            return i + 15
        self.getitem_path = ('a', 0)

class TestLoopWriteContainerNestListDict(TestWriteContainer):

    def set_func(self):
        if False:
            print('Hello World!')
        self.func = func_loop_write_nest_list_dict

    def set_getitem_path(self):
        if False:
            while True:
                i = 10
        self.getitem_path = (0, 'a')

class TestIfElseWriteContainerDict(TestWriteContainer):

    def set_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.func = func_ifelse_write_dict

    def set_getitem_path(self):
        if False:
            while True:
                i = 10
        self.getitem_path = ('a',)

class TestIfElseWriteContainerList(TestWriteContainer):

    def set_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.func = func_ifelse_write_list

    def set_getitem_path(self):
        if False:
            return 10
        self.getitem_path = (0,)

class TestIfElseWriteContainerNestDictList(TestWriteContainer):

    def set_func(self):
        if False:
            while True:
                i = 10
        self.func = func_ifelse_write_nest_dict_list

    def set_getitem_path(self):
        if False:
            print('Hello World!')
        self.getitem_path = ('a', 0)

class TestIfElseWriteContainerNestListDict(TestWriteContainer):

    def set_func(self):
        if False:
            return 10
        self.func = func_ifelse_write_nest_list_dict

    def set_getitem_path(self):
        if False:
            i = 10
            return i + 15
        self.getitem_path = (0, 'a')
if __name__ == '__main__':
    unittest.main()
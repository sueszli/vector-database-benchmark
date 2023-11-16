from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit import sot
global_x = 1
global_y = paddle.to_tensor(2)
global_z = None
global_del_val = 1
global_dict = {}
global_list = [1, 2]
global_inline = 0

def global_func_int():
    if False:
        print('Hello World!')
    global global_x
    global_x = global_x + 1
    return global_x

def global_func_int_add():
    if False:
        for i in range(10):
            print('nop')
    global global_x
    global_x = global_x + global_x
    return global_x + global_x

def global_func_tensor_int_add(tensor_y: paddle.Tensor):
    if False:
        print('Hello World!')
    global global_x
    global_x += 1
    return global_x + tensor_y

def global_multiple_update():
    if False:
        return 10
    global global_x
    global_x = 999
    global_x = 888
    global_x = 777
    return global_x - 1

def global_func_tensor():
    if False:
        i = 10
        return i + 15
    global global_y
    global_y = global_y + global_y
    return global_y

def global_func_tensor_add():
    if False:
        while True:
            i = 10
    global global_y
    global_y = global_y + global_y
    return global_y + global_y

def global_func():
    if False:
        i = 10
        return i + 15
    global global_x
    global global_y
    global global_z
    global_z = global_x + global_y
    return global_z

def global_del_global():
    if False:
        return 10
    global global_del_val
    del global_del_val

def global_func_dict():
    if False:
        while True:
            i = 10
    global global_dict
    global_dict['key'] = 'value'
    global_dict.update({'test_key1': 'test_value2'})
    return global_dict

def global_func_control1():
    if False:
        while True:
            i = 10
    global global_dict
    if 'key' in global_dict:
        del global_dict['key']
    return global_dict

def global_func_control2():
    if False:
        while True:
            i = 10
    global global_list
    for i in range(len(global_list)):
        global_list[i] = global_list[i] + 1
    return global_list

def global_func_inline_inner_1():
    if False:
        i = 10
        return i + 15
    global global_inline
    global_func_inline_inner_2()
    global_inline += 1

def global_func_inline_inner_2():
    if False:
        i = 10
        return i + 15
    global global_inline
    global_inline += 1

def global_func_inline():
    if False:
        print('Hello World!')
    global_func_inline_inner_1()
    global global_inline
    return global_inline

class TestGlobal(TestCaseBase):

    def test_global_func_int(self):
        if False:
            for i in range(10):
                print('nop')
        global global_x
        self.assert_results_with_global_check(global_func_int, ['global_x'])
        global_x += 1
        self.assert_results_with_global_check(global_func_int, ['global_x'])
        self.assert_results_with_global_check(global_func_int_add, ['global_x'])

    def test_global_multiple_update(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_global_check(global_multiple_update, ['global_x'])

    def test_global_func_tensor_int_add(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_global_check(global_func_tensor_int_add, ['global_x'], paddle.to_tensor(1))

    def test_global_func_tensor(self):
        if False:
            print('Hello World!')
        self.assert_results_with_global_check(global_func_tensor, ['global_y'])
        self.assert_results_with_global_check(global_func_tensor_add, ['global_y'])

    def test_global_func(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_global_check(global_func, ['global_z'])
        self.assertIn('global_del_val', global_del_global.__globals__)
        sot.symbolic_translate(global_del_global)()
        self.assertNotIn('global_del_val', global_del_global.__globals__)

    def test_global_func_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_global_check(global_func_dict, ['global_dict'])
        self.assert_results_with_global_check(global_func_control1, ['global_dict'])

    def test_global_func_list(self):
        if False:
            print('Hello World!')
        self.assert_results_with_global_check(global_func_control2, ['global_list'])

    def test_global_func_inline(self):
        if False:
            while True:
                i = 10
        global global_inline
        global_inline = 0
        sot.symbolic_translate(global_func_inline)()
        self.assertEqual(global_inline, 2)
        sot.symbolic_translate(global_func_inline)()
        self.assertEqual(global_inline, 4)
if __name__ == '__main__':
    unittest.main()
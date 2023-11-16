from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit import sot
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.utils import InnerError, strict_mode_guard

def dict_setitem(x):
    if False:
        return 10
    x[0] = 1
    return x[0]

def dict_delitem(x):
    if False:
        print('Hello World!')
    del x[0]
    return x

def dict_delitem_getitem(a):
    if False:
        print('Hello World!')
    b = a[0]
    del a[0]
    b[0] = 1
    return (a, b)

def dict_nested_1(x):
    if False:
        print('Hello World!')
    x[0][0] = 42
    x[1][0] = x[0][0] + x[0][1]
    x[2] = {1: 2}
    return x

def dict_nested_2(x):
    if False:
        print('Hello World!')
    a = x[0]
    b = x[1]
    del a[0]
    a[1] = b[0]
    a[2] = b[1]
    x[1][0] = 42
    del a[1]
    return (a, b)

def list_append_int(tensor_x, list_a):
    if False:
        while True:
            i = 10
    tensor_x = tensor_x + 1
    list_a.append(12)
    return (tensor_x, list_a)

def list_append_tensor(tensor_x, list_a):
    if False:
        return 10
    tensor_x = tensor_x + 1
    list_a.append(tensor_x)
    return (tensor_x, list_a)

def list_delitem(list_a):
    if False:
        while True:
            i = 10
    del list_a[0]
    return list_a[0]

def list_extend(list_a):
    if False:
        print('Hello World!')
    list_a.extend([1, 2, 3])
    return list_a[0]

def list_nested(list_a):
    if False:
        print('Hello World!')
    inner_list = []
    inner_list.append(list_a)
    inner_list[-1].append(12)
    return 12

def list_insert(list_a):
    if False:
        print('Hello World!')
    list_a.insert(0, 1)
    return list_a[0]

def list_remove(list_a):
    if False:
        print('Hello World!')
    list_a.remove(1)
    return list_a[0]

def list_pop(list_a):
    if False:
        i = 10
        return i + 15
    list_a.pop(0)
    list_a.pop()
    list_a.pop(1)
    return list_a[0]

def list_clear(list_a):
    if False:
        while True:
            i = 10
    list_a.clear()
    return list_a

def list_sort(list_a):
    if False:
        for i in range(10):
            print('nop')
    list_a.sort()
    return list_a

def list_reverse(list_a):
    if False:
        for i in range(10):
            print('nop')
    list_a.reverse()
    return list_a

def slice_in_for_loop(x, iter_num=3):
    if False:
        i = 10
        return i + 15
    x = paddle.to_tensor(x)
    a = []
    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype='int32')
    for i in range(iter_num):
        a.append(x)
    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out

class CustomObject:

    def __init__(self):
        if False:
            return 10
        self.x = 2
        self.y = paddle.to_tensor(1)

    def object_attr_set2(self, x):
        if False:
            print('Hello World!')
        self.outputs = []
        self.outputs.append(x)
        return self.outputs

@sot.psdb.check_no_breakgraph
def object_attr_set(cus_obj, t):
    if False:
        for i in range(10):
            print('nop')
    'object side effect.'
    t = t + 1
    cus_obj.x = t
    return (t, cus_obj.x)

def object_attr_breakgraph(cus_obj, t):
    if False:
        for i in range(10):
            print('nop')
    t = t + 1
    sot.psdb.breakgraph()
    cus_obj.x = t
    sot.psdb.breakgraph()
    return (t, cus_obj.x)

@sot.psdb.check_no_breakgraph
def object_attr_tensor_del(cus_obj):
    if False:
        for i in range(10):
            print('nop')
    del cus_obj.y

@sot.psdb.check_no_breakgraph
def object_attr_int_del(cus_obj):
    if False:
        print('Hello World!')
    del cus_obj.x

def slice_list_after_change(l):
    if False:
        return 10
    l.reverse()
    sum = 0
    for (i, v) in zip(range(2), l[2:]):
        sum += v
    return sum

class TestDictSideEffect(TestCaseBase):

    def test_dict_setitem(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(dict_setitem, {0: paddle.to_tensor(0)})
        self.assert_results_with_side_effects(dict_setitem, {0: paddle.to_tensor(1)})

    def test_dict_delitem(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(dict_delitem, {0: paddle.to_tensor(0), 1: paddle.to_tensor(1)})
        self.assert_results_with_side_effects(dict_delitem, {0: paddle.to_tensor(1), 2: paddle.to_tensor(2)})

    def test_dict_delitem_getitem(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(dict_delitem_getitem, {0: {0: 1, 1: 2}})

    def test_dict_nested_1(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(dict_nested_1, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}})
        self.assert_results_with_side_effects(dict_nested_1, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}})

    def test_dict_nested_2(self):
        if False:
            return 10
        self.assert_results_with_side_effects(dict_nested_2, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}})
        self.assert_results_with_side_effects(dict_nested_2, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}})

class TestListSideEffect(TestCaseBase):

    def test_list_append(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(list_append_int, paddle.to_tensor(1), [1, 2, 3])
        self.assert_results_with_side_effects(list_append_tensor, paddle.to_tensor(2), [1, 2, 3])

    def test_list_delitem(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(list_delitem, [1, 2, 3])

    def test_list_extend(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(list_extend, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_list_insert(self):
        if False:
            return 10
        self.assert_results_with_side_effects(list_insert, [1, 2, 3])
        self.assert_results_with_side_effects(list_insert, [-1, 2, -3, 4, -5, 6, -7, 8, -9])

    def test_list_remove(self):
        if False:
            i = 10
            return i + 15
        self.assert_results_with_side_effects(list_remove, [1, 1, 1])
        self.assert_results_with_side_effects(list_remove, [0, 1, 2])
        with self.assertRaises(InnerError):
            symbolic_translate(list_remove)([0, 2, 4])

    def test_list_pop(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(list_pop, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(list_pop, [-1, 2, -3, 4, -5, 6, -7, 8, -9])

    def test_list_clear(self):
        if False:
            i = 10
            return i + 15
        self.assert_results_with_side_effects(list_clear, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(list_clear, [-1, 2, -3, 4, -5, 6, -7, 8, -9])

    def test_list_sort(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(list_sort, [2, 1, 7, 3, 4, 6])
        self.assert_results_with_side_effects(list_sort, [-1, 2, -3, 4, -5, 6, -7, 8, -9])

    def test_list_reverse(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(list_reverse, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(list_reverse, [-1, 2, -3, 4, -5, 6, -7, 8, -9])

    def test_slice_in_for_loop(self):
        if False:
            return 10
        x = 2
        with strict_mode_guard(False):
            self.assert_results_with_side_effects(slice_in_for_loop, x)

    def test_list_nested(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(list_nested, [1, 2, 3])

class TestSliceAfterChange(TestCaseBase):

    def test_slice_list_after_change(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(slice_list_after_change, [1, 2, 3, 4])
        self.assert_results_with_side_effects(slice_list_after_change, [7, 8, 9, 10])

class TestAttrSideEffect(TestCaseBase):

    def attr_check(self, func, attr_keys: list[str], cls, *inputs):
        if False:
            while True:
                i = 10
        cus_obj1 = cls()
        cus_obj2 = cls()
        sym_output = symbolic_translate(func)(cus_obj1, *inputs)
        paddle_output = func(cus_obj2, *inputs)
        for key in attr_keys:
            self.assert_nest_match(getattr(cus_obj1, key, f'__MISS_KEY__{key}'), getattr(cus_obj2, key, f'__MISS_KEY__{key}'))
        self.assert_nest_match(sym_output, paddle_output)

    def test_attr_set(self):
        if False:
            i = 10
            return i + 15
        self.attr_check(object_attr_set, ['x'], CustomObject, 5)
        self.attr_check(CustomObject.object_attr_set2, ['outputs'], CustomObject, 6)
        self.attr_check(CustomObject.object_attr_set2, ['outputs'], CustomObject, paddle.to_tensor(5))
        self.attr_check(object_attr_set, ['x'], CustomObject, paddle.to_tensor(5))

    def test_attr_del(self):
        if False:
            return 10
        self.attr_check(object_attr_tensor_del, ['y'], CustomObject)
        self.attr_check(object_attr_int_del, ['x'], CustomObject)

    def test_attr_set_breakgraph(self):
        if False:
            return 10
        self.attr_check(object_attr_breakgraph, ['x'], CustomObject, 100)
        self.attr_check(object_attr_breakgraph, ['x'], CustomObject, 1000)
if __name__ == '__main__':
    unittest.main()
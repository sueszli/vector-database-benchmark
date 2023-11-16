from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

@check_no_breakgraph
def list_getitem_int(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    x = [x, y]
    return x[0] + 1

@check_no_breakgraph
def list_getitem_tensor(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    x = [x, y]
    return x[1] + 1

@check_no_breakgraph
def list_setitem_int(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = [x, y]
    z[0] = 3
    return z

def list_setitem_tensor(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = [x, y]
    z[1] = paddle.to_tensor(3)
    return z

@check_no_breakgraph
def list_delitem_int(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = [x, y]
    del z[0]
    return z

@check_no_breakgraph
def list_delitem_tensor(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = [x, y]
    del z[1]
    return z

@check_no_breakgraph
def list_construct_from_list(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = [x, y]
    return z

@check_no_breakgraph
def list_append_int(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = [x, y]
    z.append(3)
    return z

@check_no_breakgraph
def list_append_tensor(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = [x, y]
    z.append(y)
    return z

@check_no_breakgraph
def list_clear(x: int, y: paddle.Tensor):
    if False:
        print('Hello World!')
    z = [x, y]
    z.clear()
    return z

@check_no_breakgraph
def list_copy(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = [x, y]
    a = z.copy()
    z[0] = 3
    z[1] = y + 1
    return (a, z)

@check_no_breakgraph
def list_count_int(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = [x, x, 2, 3, 1]
    return z.count(x)

def list_count_tensor(x: paddle.Tensor, y: list[paddle.Tensor]):
    if False:
        return 10
    return y.count(x)

@check_no_breakgraph
def list_extend(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = [x, y]
    a = [y, x]
    b = (x, y)
    z.extend(a)
    z.extend(b)
    return z

@check_no_breakgraph
def list_index_int(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = [x, x, 1, 2]
    return z.index(x)

def list_index_tensor(x: paddle.Tensor, y: list[paddle.Tensor]):
    if False:
        i = 10
        return i + 15
    return y.index(x)

@check_no_breakgraph
def list_insert(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = [x, y]
    z.insert(0, x)
    z.insert(3, y)
    return z

@check_no_breakgraph
def list_pop(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = [x, y]
    a = z.pop()
    b = z.pop()
    return (z, a, b)

@check_no_breakgraph
def list_remove(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = [x, x, y, y]
    z.remove(x)
    z.remove(y)
    return z

@check_no_breakgraph
def list_reverse(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = [x, x, y, y]
    z.reverse()
    return z

@check_no_breakgraph
def list_default_sort(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = [x + 2, x, x + 1]
    z.sort()
    return z

@check_no_breakgraph
def list_key_sort(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = [x + 2, x, x + 1]
    z.sort(lambda x: x)
    return z

@check_no_breakgraph
def list_reverse_sort(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = [x + 2, x, x + 1]
    z.sort(reverse=True)
    return z

@check_no_breakgraph
def list_tensor_sort(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = [y + 2, y, y + 1]
    z.sort()
    return z

@check_no_breakgraph
def list_max(x: paddle.Tensor | int, y: paddle.Tensor | int):
    if False:
        for i in range(10):
            print('nop')
    z = [x, x, y]
    return max(z)

@check_no_breakgraph
def list_tensor_max_api(x: paddle.Tensor):
    if False:
        while True:
            i = 10
    return x.max()

@check_no_breakgraph
def list_min(x: paddle.Tensor | int, y: paddle.Tensor | int):
    if False:
        while True:
            i = 10
    z = [x, x, y]
    return min(z)

@check_no_breakgraph
def list_tensor_min_api(x: paddle.Tensor):
    if False:
        return 10
    return x.min()

@check_no_breakgraph
def list_no_arguments():
    if False:
        print('Hello World!')
    l1 = list()
    l1.append(1)
    l2 = list()
    l2.append(2)
    return l1[0] + l2[0]

class TestListBasic(TestCaseBase):

    def test_list_basic(self):
        if False:
            print('Hello World!')
        self.assert_results(list_getitem_int, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(list_setitem_int, 1, paddle.to_tensor(2))

class TestListMethods(TestCaseBase):

    def test_list_setitem(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(list_setitem_tensor, 1, paddle.to_tensor(2))

    def test_list_count_and_index(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(list_count_int, 1, paddle.to_tensor(2))
        self.assert_results(list_index_int, 1, paddle.to_tensor(2))
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        self.assert_results(list_count_tensor, a, [a, b, a, b, a, b])
        self.assert_results(list_index_tensor, b, [a, b, a, b, a, b])

    def test_list_delitem(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(list_delitem_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(list_delitem_tensor, 1, paddle.to_tensor(2))

    def test_list_append(self):
        if False:
            return 10
        self.assert_results_with_side_effects(list_append_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(list_append_tensor, 1, paddle.to_tensor(2))

    def test_list_clear(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(list_clear, 1, paddle.to_tensor(2))

    def test_list_copy(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(list_copy, 1, paddle.to_tensor(2))

    def test_list_extend(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(list_extend, 1, paddle.to_tensor(2))

    def test_list_insert(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(list_insert, 1, paddle.to_tensor(2))

    def test_list_pop(self):
        if False:
            i = 10
            return i + 15
        self.assert_results_with_side_effects(list_pop, 1, paddle.to_tensor(2))

    def test_list_remove(self):
        if False:
            while True:
                i = 10
        self.assert_results_with_side_effects(list_remove, 1, paddle.to_tensor(2))

    def test_list_reverse(self):
        if False:
            return 10
        self.assert_results_with_side_effects(list_reverse, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(list_reverse, 1, paddle.to_tensor(2))

    def test_list_sort(self):
        if False:
            return 10
        self.assert_results_with_side_effects(list_default_sort, 1, paddle.to_tensor(2))

    def test_list_construct_from_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(list_construct_from_list, 1, paddle.to_tensor(2))

    def test_list_max_min(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(list_max, 1, 2)
        self.assert_results(list_min, 1, 2)
        self.assert_results(list_tensor_max_api, paddle.to_tensor([1, 2, 3]))
        self.assert_results(list_tensor_min_api, paddle.to_tensor([1, 2, 3]))

    def test_list_noargs(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(list_no_arguments)
if __name__ == '__main__':
    unittest.main()
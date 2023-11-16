import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

@check_no_breakgraph
def build_map(x: int, y: paddle.Tensor):
    if False:
        print('Hello World!')
    z = {x: y}
    return z[x] + 1

@check_no_breakgraph
def build_const_key_map(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = {1: y, 2: y + 1}
    return z[x] + 1

@check_no_breakgraph
def dict_get_item(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = {1: x, 2: y + 1}
    return (z.get(1), z.get(2))

@check_no_breakgraph
def dict_get_item_default(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = {1: x, 2: y + 1}
    return (z.get(3, 2), z.get(4, y))

@check_no_breakgraph
def dict_set_item_int(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = {1: x, 2: y + 1}
    z[1] = x * 2
    return z[1]

@check_no_breakgraph
def dict_set_item_tensor(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = {1: x, 2: y + 1}
    z[2] = y
    return z[1]

@check_no_breakgraph
def dict_update_item1(x: int, y: paddle.Tensor):
    if False:
        print('Hello World!')
    z = {1: x, 2: y + 1}
    z.update({1: x * 2, 2: y, 3: y + 2})
    return z

@check_no_breakgraph
def dict_update_item2(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = {1: x, 2: y + 1}
    z.update({1: x * 2, 2: y, 3: z[2] + 2})
    return z

@check_no_breakgraph
def dict_del_item_int(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = {1: x, 2: y + 1}
    del z[1]
    return z

@check_no_breakgraph
def dict_del_item_tensor(x: int, y: paddle.Tensor):
    if False:
        print('Hello World!')
    z = {1: x, 2: y + 1}
    del z[2]
    return z

@check_no_breakgraph
def dict_clear(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = {1: x, 2: y + 1}
    z.clear()
    return z

@check_no_breakgraph
def dict_copy(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = {1: x, 2: y + 1}
    z2 = z.copy()
    z[1] = 2
    return z2

@check_no_breakgraph
def dict_setdefault_int(x: int, y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    z = {1: x, 2: y + 1}
    a = z.setdefault(4)
    b = z.setdefault(1, 2)
    c = z.setdefault(3, 4)
    return (z, a, b, c)

@check_no_breakgraph
def dict_pop(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = {1: x, 2: y + 1, 3: y}
    a = z.pop(1)
    b = z.pop(2, 3)
    c = z.pop(4, 3)
    d = z.pop(5, y)
    return (z, a, b, c, d)

@check_no_breakgraph
def dict_popitem(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = {1: x, 2: y + 1, 3: y}
    a = z.popitem()
    return (z, a)

@check_no_breakgraph
def dict_construct_from_dict():
    if False:
        while True:
            i = 10
    x = {1: 2, 3: 4}
    d = dict(x)
    return d

@check_no_breakgraph
def dict_construct_from_list():
    if False:
        for i in range(10):
            print('nop')
    x = [[1, 2], [3, 4]]
    d = dict(x)
    return d

@check_no_breakgraph
def dict_construct_from_tuple():
    if False:
        for i in range(10):
            print('nop')
    x = ((1, 2), (3, 4))
    d = dict(x)
    return d

@check_no_breakgraph
def dict_construct_from_comprehension():
    if False:
        i = 10
        return i + 15
    z = {1: 2, 3: 4}
    d = {k: v + 1 for (k, v) in z.items()}
    return d

@check_no_breakgraph
def dict_no_arguments():
    if False:
        for i in range(10):
            print('nop')
    d1 = dict()
    d1.update({1: 2})
    d2 = dict()
    d2.update({3: 4})
    return d1[1] + d2[3]

@check_no_breakgraph
def dict_test_fromkeys(x):
    if False:
        for i in range(10):
            print('nop')
    d = dict.fromkeys(x)
    return d

@check_no_breakgraph
def dict_test_fromkeys_defalut(x, y):
    if False:
        return 10
    d = dict.fromkeys(x, y)
    return d

class TestBuildDict(TestCaseBase):

    def test_build_map(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(build_map, 1, paddle.to_tensor(2))

    def test_build_const_key_map(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(build_const_key_map, 1, paddle.to_tensor(2))

class TestDictMethods(TestCaseBase):

    def test_dict_get_item(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(dict_get_item, 1, paddle.to_tensor(2))
        self.assert_results(dict_get_item_default, 1, paddle.to_tensor(2))

    def test_dict_set_item(self):
        if False:
            i = 10
            return i + 15
        self.assert_results_with_side_effects(dict_set_item_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(dict_set_item_tensor, 1, paddle.to_tensor(2))

    def test_dict_copy(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(dict_copy, 1, paddle.to_tensor(2))

    def test_dict_update(self):
        if False:
            return 10
        self.assert_results_with_side_effects(dict_update_item1, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(dict_update_item2, 1, paddle.to_tensor(2))

    def test_dict_setdefault(self):
        if False:
            i = 10
            return i + 15
        self.assert_results_with_side_effects(dict_setdefault_int, 1, paddle.to_tensor(2))

    def test_dict_del_item(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(dict_del_item_int, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(dict_del_item_tensor, 1, paddle.to_tensor(2))

    def test_dict_clear(self):
        if False:
            print('Hello World!')
        self.assert_results_with_side_effects(dict_clear, 1, paddle.to_tensor(2))

    def test_dict_pop(self):
        if False:
            return 10
        self.assert_results_with_side_effects(dict_pop, 1, paddle.to_tensor(2))

    def test_dict_popitem(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results_with_side_effects(dict_popitem, 1, paddle.to_tensor(2))

    def test_construct(self):
        if False:
            return 10
        self.assert_results(dict_construct_from_dict)
        self.assert_results(dict_construct_from_list)
        self.assert_results(dict_construct_from_tuple)
        self.assert_results(dict_construct_from_comprehension)

    def test_dict_noargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(dict_no_arguments)

    def test_dict_fromkeys(self):
        if False:
            print('Hello World!')
        self.assert_results(dict_test_fromkeys, (1, 2, 3, 4))
        self.assert_results(dict_test_fromkeys, [1, 2, 3, 4])
        self.assert_results(dict_test_fromkeys_defalut, (1, 2, 3, 4), 1)
        self.assert_results(dict_test_fromkeys_defalut, (1, 2, 3, 4), paddle.to_tensor(1))
        self.assert_results(dict_test_fromkeys_defalut, [1, 2, 3, 4], 1)
        self.assert_results(dict_test_fromkeys_defalut, [1, 2, 3, 4], paddle.to_tensor(1))
if __name__ == '__main__':
    unittest.main()
from __future__ import annotations
import unittest
from typing import Iterable
from test_case_base import TestCaseBase
from paddle.jit import sot
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard

def double_num(num: float | int):
    if False:
        while True:
            i = 10
    return num * 2

def double_num_with_breakgraph(num: float | int):
    if False:
        i = 10
        return i + 15
    sot.psdb.breakgraph()
    return num * 2

@check_no_breakgraph
def test_map_list(x: list):
    if False:
        print('Hello World!')
    return list(map(double_num, x))

@check_no_breakgraph
def test_map_list_comprehension(x: list):
    if False:
        while True:
            i = 10
    return [i for i in map(double_num, x)]

@check_no_breakgraph
def test_map_tuple(x: tuple):
    if False:
        for i in range(10):
            print('nop')
    return tuple(map(double_num, x))

@check_no_breakgraph
def test_map_tuple_comprehension(x: tuple):
    if False:
        return 10
    return [i for i in map(double_num, x)]

@check_no_breakgraph
def test_map_range(x: Iterable):
    if False:
        while True:
            i = 10
    return list(map(double_num, x))

@check_no_breakgraph
def test_map_range_comprehension(x: Iterable):
    if False:
        while True:
            i = 10
    return [i for i in map(double_num, x)]

def add_dict_prefix(key: str):
    if False:
        return 10
    return f'dict_{key}'

@check_no_breakgraph
def test_map_dict(x: dict):
    if False:
        return 10
    return list(map(add_dict_prefix, x))

@check_no_breakgraph
def test_map_dict_comprehension(x: dict):
    if False:
        while True:
            i = 10
    return [i for i in map(add_dict_prefix, x)]

def test_map_list_with_breakgraph(x: list):
    if False:
        print('Hello World!')
    return list(map(double_num_with_breakgraph, x))

@check_no_breakgraph
def test_map_unpack(x: list):
    if False:
        print('Hello World!')
    (a, b, c, d) = map(double_num, x)
    return (a, b, c, d)

@check_no_breakgraph
def test_map_for_loop(x: list):
    if False:
        return 10
    res = 0
    for i in map(double_num, x):
        res += i
    return res

class TestMap(TestCaseBase):

    def test_map(self):
        if False:
            while True:
                i = 10
        self.assert_results(test_map_list, [1, 2, 3, 4])
        self.assert_results(test_map_tuple, (1, 2, 3, 4))
        self.assert_results(test_map_range, range(5))
        self.assert_results(test_map_dict, {'a': 1, 'b': 2, 'c': 3})

    def test_map_comprehension(self):
        if False:
            while True:
                i = 10
        self.assert_results(test_map_list_comprehension, [1, 2, 3, 4])
        self.assert_results(test_map_tuple_comprehension, (1, 2, 3, 4))
        self.assert_results(test_map_range_comprehension, range(5))
        self.assert_results(test_map_dict_comprehension, {'a': 1, 'b': 2, 'c': 3})

    def test_map_with_breakgraph(self):
        if False:
            while True:
                i = 10
        with strict_mode_guard(False):
            self.assert_results(test_map_list_with_breakgraph, [1, 2, 3, 4])

    def test_map_unpack(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(test_map_unpack, [1, 2, 3, 4])

    def test_map_for_loop(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(test_map_for_loop, [7, 8, 9, 10])
if __name__ == '__main__':
    unittest.main()
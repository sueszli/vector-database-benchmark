from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit import sot
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.opcode_translator.executor.executor_cache import OpcodeExecutorCache
from paddle.jit.sot.utils import strict_mode_guard

def gener():
    if False:
        return 10
    yield 1
    yield 2
    yield 3

def for_list_1(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    for i in [1, 2, 3]:
        x += i
        if x > 2:
            x += 1
        else:
            x -= 1
    return x

def for_list_2(x: paddle.Tensor):
    if False:
        print('Hello World!')
    for i in [1, 2, 3]:
        x += i
        if i > 2:
            x += 1
        else:
            x -= 1
    return x

def for_dict(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    map = {1: 2, 3: 4}
    for k in map.keys():
        x += k
    for v in map.values():
        x += v
    for (k, v) in map.items():
        x += k
        x += v
    return x

def for_iter(x, it):
    if False:
        print('Hello World!')
    for item in it:
        x += item
    return x

def for_for_fallback(x, it):
    if False:
        return 10
    for i in [1, 2, 3]:
        for item in it:
            x += item
    return x

def for_break(x: paddle.Tensor, it):
    if False:
        for i in range(10):
            print('nop')
    for i in [1, 2, 3]:
        x += i
        if i == 2:
            break
    for i in it:
        x += i
        if i == 2:
            break
    return x

def for_continue(x: paddle.Tensor, it):
    if False:
        i = 10
        return i + 15
    for i in [1, 2, 3]:
        if i == 2:
            continue
        x += i
    for i in it:
        if i == 2:
            continue
        x += i
    return x

def for_enumerate_var_with_nested_range(x_array):
    if False:
        print('Hello World!')
    x = paddle.tensor.fill_constant([1], 'int32', 0)
    x_array = paddle.to_tensor(x_array)
    for (i, num) in enumerate(x_array):
        for idx in range(num):
            x = x + num
    return x

def for_create_tmp_in_loop(x, it):
    if False:
        i = 10
        return i + 15
    s = x
    for i in it:
        tmp = i
        s += tmp
    return (s, tmp)

def for_without_zero_iter(self_res_dict, output):
    if False:
        print('Hello World!')
    res_dict = {'logits': output}
    for res_key in list(self_res_dict):
        res_dict[res_key] = self_res_dict.pop(res_key)
    return res_dict

@sot.psdb.check_no_fallback
def for_reconstruct_range_iter():
    if False:
        i = 10
        return i + 15
    for i in range(3):
        sot.psdb.breakgraph()
global_var_name = None

def for_tmp_var_with_same_name_as_global_var():
    if False:
        for i in range(10):
            print('nop')
    total = 0
    for i in range(3):
        global_var_name = i + 3
        sot.psdb.breakgraph()
        total += global_var_name
    return total

def for_layer_list(layer_list, x):
    if False:
        print('Hello World!')
    for net in layer_list:
        x = net(x)
    return x

class TestForLoop(TestCaseBase):

    def test_list(self):
        if False:
            return 10
        a = paddle.to_tensor(1)
        self.assert_results(for_list_1, a)

    def test_list_with_fallback(self):
        if False:
            return 10
        a = paddle.to_tensor(1)
        self.assert_results(for_list_2, a)

    def test_dict(self):
        if False:
            while True:
                i = 10
        a = paddle.to_tensor(1)
        self.assert_results(for_dict, a)

    def test_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_for_fallback(self):
        if False:
            while True:
                i = 10
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    @strict_mode_guard(False)
    def test_for_break(self):
        if False:
            print('Hello World!')
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_break)(a, gener())
        paddle_output = for_break(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_continue(self):
        if False:
            return 10
        a = paddle.to_tensor(1)
        sym_output = symbolic_translate(for_continue)(a, gener())
        paddle_output = for_continue(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_create_var_in_loop(self):
        if False:
            i = 10
            return i + 15
        x = paddle.to_tensor(1, dtype='float32')
        a = [1, 2, 3]
        self.assert_results(for_create_tmp_in_loop, x, a)
        sym_output = symbolic_translate(for_create_tmp_in_loop)(x, iter(a))
        paddle_output = for_create_tmp_in_loop(x, iter(a))
        self.assert_nest_match(sym_output, paddle_output)

    def test_create_var_in_loop_with_same_name_as_global(self):
        if False:
            while True:
                i = 10
        self.assert_results(for_tmp_var_with_same_name_as_global_var)

    def test_for_without_zero_iter(self):
        if False:
            while True:
                i = 10
        self_res_dict = {}
        output = paddle.to_tensor(2)
        self.assert_results(for_without_zero_iter, self_res_dict, output)

    def test_reconstruct_range_iter(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(for_reconstruct_range_iter)

    def test_layer_list(self):
        if False:
            i = 10
            return i + 15
        layers = paddle.nn.LayerList()
        for i in range(5):
            layers.append(paddle.nn.Linear(5, 5))
        x = paddle.rand([5], dtype='float32')
        self.assert_results(for_layer_list, layers, x)

def run_list_comp(x):
    if False:
        while True:
            i = 10
    out = [s.chunk(2, axis=1) for s in x]
    return out

class TestListComp(TestCaseBase):

    def test_list_comp(self):
        if False:
            return 10
        x = [paddle.randn([1, 4]), paddle.randn([1, 4])]
        self.assert_results(run_list_comp, x)

def for_enumerate_cache(func_list, x):
    if False:
        return 10
    out = None
    for (idx, func) in enumerate(func_list):
        out = func(x[idx])
    return out

class TestEnumerateCache(TestCaseBase):

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        func_list = [paddle.nn.Linear(10, 10)]
        x = [paddle.randn([5, 10])]
        out = symbolic_translate(for_enumerate_cache)(func_list, x)
        out = symbolic_translate(for_enumerate_cache)(func_list, x)
        self.assert_nest_match(OpcodeExecutorCache().translate_count, 1)

def undefined_var_case_0():
    if False:
        return 10
    for i in [1, 2]:
        sot.psdb.breakgraph()
        zzz = i
    zzz = zzz + 1
    return zzz

def undefined_var_case_1():
    if False:
        print('Hello World!')
    for i in [1, 2]:
        sot.psdb.breakgraph()
        aaa = i
    for i in [1, 3]:
        zzz = i
    zzz = zzz + 1
    return zzz

class TestUndefinedVarInRiskyCodes(TestCaseBase):

    def test_undefined_var_case_0(self):
        if False:
            return 10
        self.assert_results(undefined_var_case_0)

    def test_undefined_var_case_1(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(undefined_var_case_1)
if __name__ == '__main__':
    unittest.main()
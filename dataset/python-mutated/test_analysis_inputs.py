from __future__ import annotations
import inspect
import sys
import unittest
import paddle
from paddle.jit.sot.opcode_translator.instruction_utils import analysis_inputs, calc_offset_from_bytecode_offset, get_instructions

def assert_inputs_equals(instruction_offset: int, expected_inputs: set[str]):
    if False:
        while True:
            i = 10
    current_frame = inspect.currentframe()
    assert current_frame is not None
    test_frame = current_frame.f_back
    assert test_frame is not None
    instructions = get_instructions(test_frame.f_code)
    current_instr_idx = calc_offset_from_bytecode_offset(test_frame.f_lasti + 2, instructions)
    actual_inputs = analysis_inputs(instructions, current_instr_idx + instruction_offset)
    assert set(actual_inputs) == expected_inputs, f'actual_inputs: {actual_inputs}, expected_inputs: {expected_inputs}'

def case1(x):
    if False:
        print('Hello World!')
    m = x + 1
    n = x + 2
    assert_inputs_equals(0, {'x', 'n'})
    y = x + 2
    assert_inputs_equals(0, {'n'})
    return n

def case2(x):
    if False:
        print('Hello World!')
    x = x + 1
    assert_inputs_equals(0, {'x'})
    y = x + 3
    z = x + y
    assert_inputs_equals(0, {'x'})
    x += 1
    m = x + 1
    n = x + m
    assert_inputs_equals(0, set())
    return 1

def case3(x):
    if False:
        return 10
    y = x + 1
    assert_inputs_equals(0, {'x'})
    if x:
        z = 1
    else:
        z = 2
    return z

def case4(x):
    if False:
        return 10
    y = x + 1
    assert_inputs_equals(0, {'x', 'y'})
    if x:
        z = y
    else:
        z = x
    return z

def case5(x):
    if False:
        while True:
            i = 10
    y = x + 1
    z = x + 2
    assert_inputs_equals(0, {'z'})
    if z:
        a = 1
    else:
        b = 2
    return z

def case6(x):
    if False:
        return 10
    y = x + 1
    z = x + 2
    assert_inputs_equals(0, {'a', 'z'})
    if z:
        a = 1
    else:
        a += 1
    return z

def case7(x):
    if False:
        for i in range(10):
            print('nop')
    y = x + 1
    z = x + 2
    assert_inputs_equals(0, {'a', 'z'})
    if not z:
        a += 1
    else:
        a = 1
    return z

def breakgraph_api(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def normal_api(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def case8(x):
    if False:
        print('Hello World!')
    x = normal_api(x)
    assert_inputs_equals(0, {'x'})
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x
case9_offset = -9 if sys.version_info >= (3, 11) else -7

def case9(x):
    if False:
        while True:
            i = 10
    x = breakgraph_api(x)
    assert_inputs_equals(case9_offset, set())
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x

def case10(x):
    if False:
        for i in range(10):
            print('nop')
    assert_inputs_equals(0, {'x', 'y'})
    for i in range(x):
        y = i
        z = y
    return y + 1

def case11(x):
    if False:
        while True:
            i = 10
    y = x + 1
    z = x + 2
    assert_inputs_equals(0, {'a', 'y', 'z'})
    if z:
        if not y:
            a += 1
        else:
            a = 2
    elif y:
        a = 1
    else:
        a += 1
    return z

def case12(x):
    if False:
        print('Hello World!')
    y = x + 1
    z = x + 2
    assert_inputs_equals(0, {'a', 'y', 'z'})
    if z:
        if y:
            a = 2
        else:
            a += 2
    elif y:
        a += 1
    else:
        a = 1
    return z

class TestAnalysisInputs(unittest.TestCase):

    def test_case1(self):
        if False:
            for i in range(10):
                print('nop')
        case1(paddle.to_tensor([1]))

    def test_case2(self):
        if False:
            i = 10
            return i + 15
        case2(paddle.to_tensor([2]))

    def test_case3(self):
        if False:
            i = 10
            return i + 15
        case3(paddle.to_tensor([3]))

    def test_case4(self):
        if False:
            print('Hello World!')
        case4(paddle.to_tensor([4]))

    def test_case5(self):
        if False:
            print('Hello World!')
        case5(paddle.to_tensor([5]))

    def test_case6(self):
        if False:
            i = 10
            return i + 15
        case6(paddle.to_tensor([6]))

    def test_case7(self):
        if False:
            print('Hello World!')
        case7(paddle.to_tensor([7]))

    def test_case8(self):
        if False:
            i = 10
            return i + 15
        case8(paddle.to_tensor([8]))

    def test_case9(self):
        if False:
            for i in range(10):
                print('nop')
        case9(paddle.to_tensor([9]))

    def test_case10(self):
        if False:
            for i in range(10):
                print('nop')
        case10(paddle.to_tensor([10]))

    def test_case11(self):
        if False:
            print('Hello World!')
        case11(paddle.to_tensor([11]))

    def test_case12(self):
        if False:
            i = 10
            return i + 15
        case12(paddle.to_tensor([12]))
if __name__ == '__main__':
    unittest.main()
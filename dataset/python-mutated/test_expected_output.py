"""
'Golden master' tests for the ghostwriter.

To update the recorded outputs, run `pytest --hypothesis-update-outputs ...`.
"""
import ast
import base64
import builtins
import collections.abc
import operator
import pathlib
import re
import sys
from typing import Optional, Sequence, Union
import numpy
import pytest
from example_code.future_annotations import add_custom_classes, invalid_types, merge_dicts
import hypothesis
from hypothesis.extra import ghostwriter
from hypothesis.utils.conventions import not_set

@pytest.fixture
def update_recorded_outputs(request):
    if False:
        print('Hello World!')
    return request.config.getoption('--hypothesis-update-outputs')

def get_recorded(name, actual=''):
    if False:
        return 10
    file_ = pathlib.Path(__file__).parent / 'recorded' / f'{name}.txt'
    if actual:
        file_.write_text(actual, encoding='utf-8')
    return file_.read_text(encoding='utf-8')

def timsort(seq: Sequence[int]) -> Sequence[int]:
    if False:
        i = 10
        return i + 15
    return sorted(seq)

def with_docstring(a, b, c, d=int, e=lambda x: f'xx{x}xx') -> None:
    if False:
        i = 10
        return i + 15
    'Demonstrates parsing params from the docstring\n\n    :param a: sphinx docstring style\n    :type a: sequence of integers\n\n    b (list, tuple, or None): Google docstring style\n\n    c : {"foo", "bar", or None}\n        Numpy docstring style\n    '

class A_Class:

    @classmethod
    def a_classmethod(cls, arg: int):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def a_staticmethod(arg: int):
        if False:
            for i in range(10):
                print('nop')
        pass

def add(a: float, b: float) -> float:
    if False:
        print('Hello World!')
    return a + b

def divide(a: int, b: int) -> float:
    if False:
        for i in range(10):
            print('nop')
    'This is a RST-style docstring for `divide`.\n\n    :raises ZeroDivisionError: if b == 0\n    '
    return a / b

def optional_parameter(a: float, b: Optional[float]) -> float:
    if False:
        print('Hello World!')
    return optional_union_parameter(a, b)

def optional_union_parameter(a: float, b: Optional[Union[float, int]]) -> float:
    if False:
        while True:
            i = 10
    return a if b is None else a + b
if sys.version_info[:2] >= (3, 10):

    def union_sequence_parameter(items: Sequence[float | int]) -> float:
        if False:
            while True:
                i = 10
        return sum(items)
else:

    def union_sequence_parameter(items: Sequence[Union[float, int]]) -> float:
        if False:
            for i in range(10):
                print('nop')
        return sum(items)
if sys.version_info[:2] >= (3, 9):
    CollectionsSequence = collections.abc.Sequence
else:
    CollectionsSequence = Sequence

def sequence_from_collections(items: CollectionsSequence[int]) -> int:
    if False:
        while True:
            i = 10
    return min(items)

@pytest.mark.parametrize('data', [('fuzz_sorted', ghostwriter.fuzz(sorted)), ('fuzz_sorted_with_annotations', ghostwriter.fuzz(sorted, annotate=True)), ('fuzz_with_docstring', ghostwriter.fuzz(with_docstring)), ('fuzz_classmethod', ghostwriter.fuzz(A_Class.a_classmethod)), ('fuzz_staticmethod', ghostwriter.fuzz(A_Class.a_staticmethod)), ('fuzz_ufunc', ghostwriter.fuzz(numpy.add)), ('magic_gufunc', ghostwriter.magic(numpy.matmul)), pytest.param(('optional_parameter', ghostwriter.magic(optional_parameter)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 9)')), pytest.param(('optional_parameter_pre_py_3_9', ghostwriter.magic(optional_parameter)), marks=pytest.mark.skipif('sys.version_info[:2] >= (3, 9)')), ('optional_union_parameter', ghostwriter.magic(optional_union_parameter)), ('union_sequence_parameter', ghostwriter.magic(union_sequence_parameter)), pytest.param(('sequence_from_collections', ghostwriter.magic(sequence_from_collections)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 9)')), pytest.param(('add_custom_classes', ghostwriter.magic(add_custom_classes)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')), pytest.param(('merge_dicts', ghostwriter.magic(merge_dicts)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')), pytest.param(('invalid_types', ghostwriter.magic(invalid_types)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')), ('magic_base64_roundtrip', ghostwriter.magic(base64.b64encode)), ('magic_base64_roundtrip_with_annotations', ghostwriter.magic(base64.b64encode, annotate=True)), ('re_compile', ghostwriter.fuzz(re.compile)), ('re_compile_except', ghostwriter.fuzz(re.compile, except_=re.error).replace('import sre_constants\n', '').replace('sre_constants.', 're.')), ('re_compile_unittest', ghostwriter.fuzz(re.compile, style='unittest')), pytest.param(('base64_magic', ghostwriter.magic(base64)), marks=pytest.mark.skipif('sys.version_info[:2] >= (3, 10)')), ('sorted_idempotent', ghostwriter.idempotent(sorted)), ('timsort_idempotent', ghostwriter.idempotent(timsort)), ('timsort_idempotent_asserts', ghostwriter.idempotent(timsort, except_=AssertionError)), ('eval_equivalent', ghostwriter.equivalent(eval, ast.literal_eval)), ('sorted_self_equivalent', ghostwriter.equivalent(sorted, sorted, sorted)), ('sorted_self_equivalent_with_annotations', ghostwriter.equivalent(sorted, sorted, sorted, annotate=True)), ('addition_op_magic', ghostwriter.magic(add)), ('addition_op_multimagic', ghostwriter.magic(add, operator.add, numpy.add)), ('division_fuzz_error_handler', ghostwriter.fuzz(divide)), ('division_binop_error_handler', ghostwriter.binary_operation(divide, identity=1)), ('division_roundtrip_error_handler', ghostwriter.roundtrip(divide, operator.mul)), ('division_roundtrip_error_handler_without_annotations', ghostwriter.roundtrip(divide, operator.mul, annotate=False)), ('division_roundtrip_arithmeticerror_handler', ghostwriter.roundtrip(divide, operator.mul, except_=ArithmeticError)), ('division_roundtrip_typeerror_handler', ghostwriter.roundtrip(divide, operator.mul, except_=TypeError)), ('division_operator', ghostwriter.binary_operation(operator.truediv, associative=False, commutative=False)), ('division_operator_with_annotations', ghostwriter.binary_operation(operator.truediv, associative=False, commutative=False, annotate=True)), ('multiplication_operator', ghostwriter.binary_operation(operator.mul, identity=1, distributes_over=operator.add)), ('multiplication_operator_unittest', ghostwriter.binary_operation(operator.mul, identity=1, distributes_over=operator.add, style='unittest')), ('sorted_self_error_equivalent_simple', ghostwriter.equivalent(sorted, sorted, allow_same_errors=True)), ('sorted_self_error_equivalent_threefuncs', ghostwriter.equivalent(sorted, sorted, sorted, allow_same_errors=True)), ('sorted_self_error_equivalent_1error', ghostwriter.equivalent(sorted, sorted, allow_same_errors=True, except_=ValueError)), ('sorted_self_error_equivalent_2error_unittest', ghostwriter.equivalent(sorted, sorted, allow_same_errors=True, except_=(TypeError, ValueError), style='unittest')), ('magic_class', ghostwriter.magic(A_Class)), pytest.param(('magic_builtins', ghostwriter.magic(builtins)), marks=[pytest.mark.skipif(sys.version_info[:2] not in [(3, 8), (3, 9)], reason='compile arg new in 3.8, aiter and anext new in 3.10')])], ids=lambda x: x[0])
def test_ghostwriter_example_outputs(update_recorded_outputs, data):
    if False:
        i = 10
        return i + 15
    (name, actual) = data
    expected = get_recorded(name, actual * update_recorded_outputs)
    assert actual == expected
    exec(expected, {})

def test_ghostwriter_on_hypothesis(update_recorded_outputs):
    if False:
        for i in range(10):
            print('nop')
    actual = ghostwriter.magic(hypothesis).replace('Strategy[+Ex]', 'Strategy')
    expected = get_recorded('hypothesis_module_magic', actual * update_recorded_outputs)
    if sys.version_info[:2] == (3, 10):
        assert actual == expected
    exec(expected, {'not_set': not_set})
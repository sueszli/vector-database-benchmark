import pytest
from vyper.ast import parse_to_ast
from vyper.exceptions import ArgumentException, ImmutableViolation, StateAccessViolation, TypeMismatch
from vyper.semantics.analysis import validate_semantics

def test_modify_iterator_function_outside_loop(namespace):
    if False:
        while True:
            i = 10
    code = '\n\na: uint256[3]\n\n@internal\ndef foo():\n    self.a[0] = 1\n\n@internal\ndef bar():\n    self.foo()\n    for i in self.a:\n        pass\n    '
    vyper_module = parse_to_ast(code)
    validate_semantics(vyper_module, {})

def test_pass_memory_var_to_other_function(namespace):
    if False:
        while True:
            i = 10
    code = '\n\n@internal\ndef foo(a: uint256[3]) -> uint256[3]:\n    b: uint256[3] = a\n    b[1] = 42\n    return b\n\n\n@internal\ndef bar():\n    a: uint256[3] = [1,2,3]\n    for i in a:\n        self.foo(a)\n    '
    vyper_module = parse_to_ast(code)
    validate_semantics(vyper_module, {})

def test_modify_iterator(namespace):
    if False:
        return 10
    code = '\n\na: uint256[3]\n\n@internal\ndef bar():\n    for i in self.a:\n        self.a[0] = 1\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(ImmutableViolation):
        validate_semantics(vyper_module, {})

def test_bad_keywords(namespace):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\n@internal\ndef bar(n: uint256):\n    x: uint256 = 0\n    for i in range(n, boundddd=10):\n        x += i\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(ArgumentException):
        validate_semantics(vyper_module, {})

def test_bad_bound(namespace):
    if False:
        i = 10
        return i + 15
    code = '\n\n@internal\ndef bar(n: uint256):\n    x: uint256 = 0\n    for i in range(n, bound=n):\n        x += i\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(StateAccessViolation):
        validate_semantics(vyper_module, {})

def test_modify_iterator_function_call(namespace):
    if False:
        for i in range(10):
            print('nop')
    code = '\n\na: uint256[3]\n\n@internal\ndef foo():\n    self.a[0] = 1\n\n@internal\ndef bar():\n    for i in self.a:\n        self.foo()\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(ImmutableViolation):
        validate_semantics(vyper_module, {})

def test_modify_iterator_recursive_function_call(namespace):
    if False:
        return 10
    code = '\n\na: uint256[3]\n\n@internal\ndef foo():\n    self.a[0] = 1\n\n@internal\ndef bar():\n    self.foo()\n\n@internal\ndef baz():\n    for i in self.a:\n        self.bar()\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(ImmutableViolation):
        validate_semantics(vyper_module, {})
iterator_inference_codes = ['\n@external\ndef main():\n    for j in range(3):\n        x: uint256 = j\n        y: uint16 = j\n    ', '\n@external\ndef foo():\n    for i in [1]:\n        a:uint256 = i\n        b:uint16 = i\n    ', '\n@external\ndef foo():\n    for i in [1]:\n        for j in [1]:\n            a:uint256 = i\n        b:uint16 = i\n    ', '\n@external\ndef foo():\n    for i in [1,2,3]:\n        for j in [1,2,3]:\n            b:uint256 = j + i\n        c:uint16 = i\n    ']

@pytest.mark.parametrize('code', iterator_inference_codes)
def test_iterator_type_inference_checker(namespace, code):
    if False:
        i = 10
        return i + 15
    vyper_module = parse_to_ast(code)
    with pytest.raises(TypeMismatch):
        validate_semantics(vyper_module, {})
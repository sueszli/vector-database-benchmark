import pytest
from vyper.ast import parse_to_ast
from vyper.exceptions import CallViolation, StructureException
from vyper.semantics.analysis import validate_semantics
from vyper.semantics.analysis.module import ModuleAnalyzer

def test_self_function_call(namespace):
    if False:
        while True:
            i = 10
    code = '\n@internal\ndef foo():\n    self.foo()\n    '
    vyper_module = parse_to_ast(code)
    with namespace.enter_scope():
        with pytest.raises(CallViolation):
            ModuleAnalyzer(vyper_module, {}, namespace)

def test_cyclic_function_call(namespace):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@internal\ndef foo():\n    self.bar()\n\n@internal\ndef bar():\n    self.foo()\n    '
    vyper_module = parse_to_ast(code)
    with namespace.enter_scope():
        with pytest.raises(CallViolation):
            ModuleAnalyzer(vyper_module, {}, namespace)

def test_multi_cyclic_function_call(namespace):
    if False:
        print('Hello World!')
    code = '\n@internal\ndef foo():\n    self.bar()\n\n@internal\ndef bar():\n    self.baz()\n\n@internal\ndef baz():\n    self.potato()\n\n@internal\ndef potato():\n    self.foo()\n    '
    vyper_module = parse_to_ast(code)
    with namespace.enter_scope():
        with pytest.raises(CallViolation):
            ModuleAnalyzer(vyper_module, {}, namespace)

def test_global_ann_assign_callable_no_crash():
    if False:
        i = 10
        return i + 15
    code = '\nbalanceOf: public(HashMap[address, uint256])\n\n@internal\ndef foo(to : address):\n    self.balanceOf(to)\n    '
    vyper_module = parse_to_ast(code)
    with pytest.raises(StructureException) as excinfo:
        validate_semantics(vyper_module, {})
    assert excinfo.value.message == 'Value is not callable'
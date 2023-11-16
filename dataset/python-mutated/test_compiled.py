from textwrap import dedent
import sys
import math
from collections import Counter
from datetime import datetime
import pytest
from jedi.inference import compiled
from jedi.inference.compiled.access import DirectObjectAccess
from jedi.inference.gradual.conversion import _stub_to_python_value_set
from jedi.inference.syntax_tree import _infer_comparison_part

def test_simple(inference_state, environment):
    if False:
        return 10
    obj = compiled.create_simple_object(inference_state, '_str_')
    (upper,) = obj.py__getattribute__('upper')
    objs = list(upper.execute_with_values())
    assert len(objs) == 1
    assert objs[0].name.string_name == 'str'

def test_builtin_loading(inference_state):
    if False:
        print('Hello World!')
    (string,) = inference_state.builtins_module.py__getattribute__('str')
    (from_name,) = string.py__getattribute__('__init__')
    assert from_name.tree_node
    assert not from_name.py__doc__()

def test_next_docstr(inference_state, environment):
    if False:
        while True:
            i = 10
    if environment.version_info[:2] != sys.version_info[:2]:
        pytest.skip()
    next_ = compiled.builtin_from_name(inference_state, 'next')
    assert next_.tree_node is not None
    assert next_.py__doc__() == ''
    for non_stub in _stub_to_python_value_set(next_):
        assert non_stub.py__doc__() == next.__doc__

def test_parse_function_doc_illegal_docstr():
    if False:
        return 10
    docstr = "\n    test_func(o\n\n    doesn't have a closing bracket.\n    "
    assert ('', '') == compiled.value._parse_function_doc(docstr)

def test_doc(inference_state):
    if False:
        i = 10
        return i + 15
    "\n    Even CompiledValue docs always return empty docstrings - not None, that's\n    just a Jedi API definition.\n    "
    str_ = compiled.create_simple_object(inference_state, '')
    (obj,) = str_.py__getattribute__('__getnewargs__')
    assert obj.py__doc__() == ''

def test_string_literals(Script, environment):
    if False:
        i = 10
        return i + 15

    def typ(string):
        if False:
            while True:
                i = 10
        d = Script('a = %s; a' % string).infer()[0]
        return d.name
    assert typ('""') == 'str'
    assert typ('r""') == 'str'
    assert typ('br""') == 'bytes'
    assert typ('b""') == 'bytes'
    assert typ('u""') == 'str'

def test_method_completion(Script, environment):
    if False:
        while True:
            i = 10
    code = dedent('\n    class Foo:\n        def bar(self):\n            pass\n\n    foo = Foo()\n    foo.bar.__func__')
    assert [c.name for c in Script(code).complete()] == ['__func__']

def test_time_docstring(Script):
    if False:
        return 10
    import time
    (comp,) = Script('import time\ntime.sleep').complete()
    assert comp.docstring(raw=True) == time.sleep.__doc__
    expected = 'sleep(secs: float) -> None\n\n' + time.sleep.__doc__
    assert comp.docstring() == expected

def test_dict_values(Script, environment):
    if False:
        for i in range(10):
            print('nop')
    assert Script('import sys\nsys.modules["alshdb;lasdhf"]').infer()

def test_getitem_on_none(Script):
    if False:
        print('Hello World!')
    script = Script('None[1j]')
    assert not script.infer()
    (issue,) = script._inference_state.analysis
    assert issue.name == 'type-error-not-subscriptable'

def _return_int():
    if False:
        return 10
    return 1

@pytest.mark.parametrize('attribute, expected_name, expected_parent', [('x', 'int', 'builtins'), ('y', 'int', 'builtins'), ('z', 'bool', 'builtins'), ('cos', 'cos', 'math'), ('dec', 'Decimal', 'decimal'), ('dt', 'datetime', 'datetime'), ('ret_int', '_return_int', 'test.test_inference.test_compiled')])
def test_parent_context(same_process_inference_state, attribute, expected_name, expected_parent):
    if False:
        for i in range(10):
            print('nop')
    import decimal

    class C:
        x = 1
        y = int
        z = True
        cos = math.cos
        dec = decimal.Decimal(1)
        dt = datetime(2000, 1, 1)
        ret_int = _return_int
    o = compiled.CompiledValue(same_process_inference_state, DirectObjectAccess(same_process_inference_state, C))
    (x,) = o.py__getattribute__(attribute)
    assert x.py__name__() == expected_name
    module_name = x.parent_context.py__name__()
    assert module_name == expected_parent
    assert x.parent_context.parent_context is None

@pytest.mark.parametrize('obj, expected_names', [('', ['str']), (str, ['str']), (''.upper, ['str', 'upper']), (str.upper, ['str', 'upper']), (math.cos, ['cos']), (Counter, ['Counter']), (Counter(''), ['Counter']), (Counter.most_common, ['Counter', 'most_common']), (Counter('').most_common, ['Counter', 'most_common'])])
def test_qualified_names(same_process_inference_state, obj, expected_names):
    if False:
        return 10
    o = compiled.CompiledValue(same_process_inference_state, DirectObjectAccess(same_process_inference_state, obj))
    assert o.get_qualified_names() == tuple(expected_names)

def test_operation(Script, inference_state, create_compiled_object):
    if False:
        while True:
            i = 10
    b = create_compiled_object(bool)
    (false, true) = _infer_comparison_part(inference_state, b.parent_context, left=list(b.execute_with_values())[0], operator='is not', right=b)
    assert false.py__name__() == 'bool'
    assert true.py__name__() == 'bool'
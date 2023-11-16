import os
import pytest
from parso.utils import PythonVersionInfo
from jedi.inference.gradual import typeshed
from jedi.inference.value import TreeInstance, BoundMethod, FunctionValue, MethodValue, ClassValue
from jedi.inference.names import StubName
TYPESHED_PYTHON3 = os.path.join(typeshed.TYPESHED_PATH, 'stdlib', '3')

def test_get_typeshed_directories():
    if False:
        return 10

    def get_dirs(version_info):
        if False:
            return 10
        return {p.path.replace(str(typeshed.TYPESHED_PATH), '').lstrip(os.path.sep) for p in typeshed._get_typeshed_directories(version_info)}

    def transform(set_):
        if False:
            while True:
                i = 10
        return {x.replace('/', os.path.sep) for x in set_}
    dirs = get_dirs(PythonVersionInfo(3, 7))
    assert dirs == transform({'stdlib/2and3', 'stdlib/3', 'stdlib/3.7', 'third_party/2and3', 'third_party/3', 'third_party/3.7'})

def test_get_stub_files():
    if False:
        while True:
            i = 10
    map_ = typeshed._create_stub_map(typeshed.PathInfo(TYPESHED_PYTHON3, is_third_party=False))
    assert map_['functools'].path == os.path.join(TYPESHED_PYTHON3, 'functools.pyi')

def test_function(Script, environment):
    if False:
        while True:
            i = 10
    code = 'import threading; threading.current_thread'
    (def_,) = Script(code).infer()
    value = def_._name._value
    assert isinstance(value, FunctionValue), value
    (def_,) = Script(code + '()').infer()
    value = def_._name._value
    assert isinstance(value, TreeInstance)
    (def_,) = Script('import threading; threading.Thread').infer()
    assert isinstance(def_._name._value, ClassValue), def_

def test_keywords_variable(Script):
    if False:
        print('Hello World!')
    code = 'import keyword; keyword.kwlist'
    for seq in Script(code).infer():
        assert seq.name == 'Sequence'
        (stub_seq,) = seq.goto(only_stubs=True)
        assert str(stub_seq.module_path).startswith(str(typeshed.TYPESHED_PATH))

def test_class(Script):
    if False:
        for i in range(10):
            print('nop')
    (def_,) = Script('import threading; threading.Thread').infer()
    value = def_._name._value
    assert isinstance(value, ClassValue), value

def test_instance(Script):
    if False:
        for i in range(10):
            print('nop')
    (def_,) = Script('import threading; threading.Thread()').infer()
    value = def_._name._value
    assert isinstance(value, TreeInstance)

def test_class_function(Script):
    if False:
        for i in range(10):
            print('nop')
    (def_,) = Script('import threading; threading.Thread.getName').infer()
    value = def_._name._value
    assert isinstance(value, MethodValue), value

def test_method(Script):
    if False:
        return 10
    code = 'import threading; threading.Thread().getName'
    (def_,) = Script(code).infer()
    value = def_._name._value
    assert isinstance(value, BoundMethod), value
    assert isinstance(value._wrapped_value, MethodValue), value
    (def_,) = Script(code + '()').infer()
    value = def_._name._value
    assert isinstance(value, TreeInstance)
    assert value.class_value.py__name__() == 'str'

def test_sys_exc_info(Script):
    if False:
        while True:
            i = 10
    code = 'import sys; sys.exc_info()'
    (none, def_) = Script(code + '[1]').infer()
    assert def_.name == 'BaseException'
    assert def_.module_path == typeshed.TYPESHED_PATH.joinpath('stdlib', '3', 'builtins.pyi')
    assert def_.type == 'instance'
    assert none.name == 'NoneType'
    assert none.module_path is None
    (none, def_) = Script(code + '[0]').infer()
    assert def_.name == 'BaseException'
    assert def_.type == 'class'

def test_sys_getwindowsversion(Script, environment):
    if False:
        while True:
            i = 10
    definitions = Script('import sys; sys.getwindowsversion().major').infer()
    (def_,) = definitions
    assert def_.name == 'int'

def test_sys_hexversion(Script):
    if False:
        print('Hello World!')
    script = Script('import sys; sys.hexversion')
    (def_,) = script.complete()
    assert isinstance(def_._name, StubName), def_._name
    assert str(def_.module_path).startswith(str(typeshed.TYPESHED_PATH))
    (def_,) = script.infer()
    assert def_.name == 'int'

def test_math(Script):
    if False:
        for i in range(10):
            print('nop')
    (def_,) = Script('import math; math.acos()').infer()
    assert def_.name == 'float'
    value = def_._name._value
    assert value

def test_type_var(Script):
    if False:
        while True:
            i = 10
    (def_,) = Script('import typing; T = typing.TypeVar("T1")').infer()
    assert def_.name == 'TypeVar'
    assert def_.description == 'class TypeVar'

@pytest.mark.parametrize('code, full_name', (('import math', 'math'), ('from math import cos', 'math.cos')))
def test_math_is_stub(Script, code, full_name):
    if False:
        while True:
            i = 10
    s = Script(code)
    (cos,) = s.infer()
    wanted = ('typeshed', 'stdlib', '2and3', 'math.pyi')
    assert cos.module_path.parts[-4:] == wanted
    assert cos.is_stub() is True
    assert cos.goto(only_stubs=True) == [cos]
    assert cos.full_name == full_name
    (cos,) = s.goto()
    assert cos.module_path.parts[-4:] == wanted
    assert cos.goto(only_stubs=True) == [cos]
    assert cos.is_stub() is True
    assert cos.full_name == full_name

def test_goto_stubs(Script):
    if False:
        i = 10
        return i + 15
    s = Script('import os; os')
    (os_module,) = s.infer()
    assert os_module.full_name == 'os'
    assert os_module.is_stub() is False
    (stub,) = os_module.goto(only_stubs=True)
    assert stub.is_stub() is True
    (os_module,) = s.goto()

def _assert_is_same(d1, d2):
    if False:
        i = 10
        return i + 15
    assert d1.name == d2.name
    assert d1.module_path == d2.module_path
    assert d1.line == d2.line
    assert d1.column == d2.column

@pytest.mark.parametrize('type_', ['goto', 'infer'])
@pytest.mark.parametrize('code', ['import os; os.walk', 'from collections import Counter; Counter', 'from collections import Counter; Counter()', 'from collections import Counter; Counter.most_common', 'from collections import Counter; Counter().most_common'])
def test_goto_stubs_on_itself(Script, code, type_):
    if False:
        while True:
            i = 10
    '\n    If goto_stubs is used on an identifier in e.g. the stdlib, we should goto\n    the stub of it.\n    '
    s = Script(code)
    if type_ == 'infer':
        (def_,) = s.infer()
    else:
        (def_,) = s.goto(follow_imports=True)
    (stub,) = def_.goto(only_stubs=True)
    script_on_source = Script(path=def_.module_path)
    if type_ == 'infer':
        (definition,) = script_on_source.infer(def_.line, def_.column)
    else:
        (definition,) = script_on_source.goto(def_.line, def_.column)
    (same_stub,) = definition.goto(only_stubs=True)
    _assert_is_same(same_stub, stub)
    _assert_is_same(definition, def_)
    assert same_stub.module_path != def_.module_path
    script_on_stub = Script(path=same_stub.module_path)
    if type_ == 'infer':
        (same_definition,) = script_on_stub.infer(same_stub.line, same_stub.column)
        (same_definition2,) = same_stub.infer()
    else:
        (same_definition,) = script_on_stub.goto(same_stub.line, same_stub.column)
        (same_definition2,) = same_stub.goto()
    _assert_is_same(same_definition, definition)
    _assert_is_same(same_definition, same_definition2)

def test_module_exists_only_as_stub(Script):
    if False:
        print('Hello World!')
    try:
        import redis
    except ImportError:
        pass
    else:
        pytest.skip('redis is already installed, it should only exist as a stub for this test')
    redis_path = os.path.join(typeshed.TYPESHED_PATH, 'third_party', '2and3', 'redis')
    assert os.path.isdir(redis_path)
    assert not Script('import redis').infer()

def test_django_exists_only_as_stub(Script):
    if False:
        for i in range(10):
            print('nop')
    try:
        import django
    except ImportError:
        pass
    else:
        pytest.skip('django is already installed, it should only exist as a stub for this test')
    assert not Script('import django').infer()
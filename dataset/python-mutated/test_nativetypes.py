import math
import pytest
from jinja2.exceptions import UndefinedError
from jinja2.nativetypes import NativeEnvironment
from jinja2.nativetypes import NativeTemplate
from jinja2.runtime import Undefined

@pytest.fixture
def env():
    if False:
        for i in range(10):
            print('nop')
    return NativeEnvironment()

def test_is_defined_native_return(env):
    if False:
        while True:
            i = 10
    t = env.from_string('{{ missing is defined }}')
    assert not t.render()

def test_undefined_native_return(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string('{{ missing }}')
    assert isinstance(t.render(), Undefined)

def test_adding_undefined_native_return(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string('{{ 3 + missing }}')
    with pytest.raises(UndefinedError):
        t.render()

def test_cast_int(env):
    if False:
        while True:
            i = 10
    t = env.from_string('{{ value|int }}')
    result = t.render(value='3')
    assert isinstance(result, int)
    assert result == 3

def test_list_add(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string('{{ a + b }}')
    result = t.render(a=['a', 'b'], b=['c', 'd'])
    assert isinstance(result, list)
    assert result == ['a', 'b', 'c', 'd']

def test_multi_expression_add(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string('{{ a }} + {{ b }}')
    result = t.render(a=['a', 'b'], b=['c', 'd'])
    assert not isinstance(result, list)
    assert result == "['a', 'b'] + ['c', 'd']"

def test_loops(env):
    if False:
        print('Hello World!')
    t = env.from_string('{% for x in value %}{{ x }}{% endfor %}')
    result = t.render(value=['a', 'b', 'c', 'd'])
    assert isinstance(result, str)
    assert result == 'abcd'

def test_loops_with_ints(env):
    if False:
        print('Hello World!')
    t = env.from_string('{% for x in value %}{{ x }}{% endfor %}')
    result = t.render(value=[1, 2, 3, 4])
    assert isinstance(result, int)
    assert result == 1234

def test_loop_look_alike(env):
    if False:
        return 10
    t = env.from_string('{% for x in value %}{{ x }}{% endfor %}')
    result = t.render(value=[1])
    assert isinstance(result, int)
    assert result == 1

@pytest.mark.parametrize(('source', 'expect'), (('{{ value }}', True), ('{{ value }}', False), ('{{ 1 == 1 }}', True), ('{{ 2 + 2 == 5 }}', False), ('{{ None is none }}', True), ("{{ '' == None }}", False)))
def test_booleans(env, source, expect):
    if False:
        print('Hello World!')
    t = env.from_string(source)
    result = t.render(value=expect)
    assert isinstance(result, bool)
    assert result is expect

def test_variable_dunder(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string('{{ x.__class__ }}')
    result = t.render(x=True)
    assert isinstance(result, type)

def test_constant_dunder(env):
    if False:
        while True:
            i = 10
    t = env.from_string('{{ true.__class__ }}')
    result = t.render()
    assert isinstance(result, type)

def test_constant_dunder_to_string(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string('{{ true.__class__|string }}')
    result = t.render()
    assert not isinstance(result, type)
    assert result in {"<type 'bool'>", "<class 'bool'>"}

def test_string_literal_var(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string("[{{ 'all' }}]")
    result = t.render()
    assert isinstance(result, str)
    assert result == '[all]'

def test_string_top_level(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string("'Jinja'")
    result = t.render()
    assert result == 'Jinja'

def test_tuple_of_variable_strings(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string("'{{ a }}', 'data', '{{ b }}', b'{{ c }}'")
    result = t.render(a=1, b=2, c='bytes')
    assert isinstance(result, tuple)
    assert result == ('1', 'data', '2', b'bytes')

def test_concat_strings_with_quotes(env):
    if False:
        for i in range(10):
            print('nop')
    t = env.from_string('--host=\'{{ host }}\' --user "{{ user }}"')
    result = t.render(host='localhost', user='Jinja')
    assert result == '--host=\'localhost\' --user "Jinja"'

def test_no_intermediate_eval(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string('0.000{{ a }}')
    result = t.render(a=7)
    assert isinstance(result, float)
    assert math.isclose(result, 0.0007)

def test_spontaneous_env():
    if False:
        i = 10
        return i + 15
    t = NativeTemplate('{{ true }}')
    assert isinstance(t.environment, NativeEnvironment)

def test_leading_spaces(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string(' {{ True }}')
    result = t.render()
    assert result == ' True'

def test_macro(env):
    if False:
        i = 10
        return i + 15
    t = env.from_string('{%- macro x() -%}{{- [1,2] -}}{%- endmacro -%}{{- x()[1] -}}')
    result = t.render()
    assert result == 2
    assert isinstance(result, int)
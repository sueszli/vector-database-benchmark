import json
import pytest
from lark.lark import Lark
from hypothesis import given
from hypothesis.errors import InvalidArgument
from hypothesis.extra.lark import from_lark
from hypothesis.strategies import data, just
from tests.common.debug import find_any
EBNF_GRAMMAR = '\n    value: dict\n         | list\n         | STRING\n         | NUMBER\n         | "true"  -> true\n         | "false" -> false\n         | "null"  -> null\n    list : "[" [value ("," value)*] "]"\n    dict : "{" [STRING ":" value ("," STRING ":" value)*] "}"\n\n    STRING : /"[a-z]*"/\n    NUMBER : /-?[1-9][0-9]*(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n\n    WS : /[ \\t\\r\\n]+/\n    %ignore WS\n'
LIST_GRAMMAR = '\nlist : "[" [NUMBER ("," NUMBER)*] "]"\nNUMBER: /[0-9]+/\n'

@given(from_lark(Lark(EBNF_GRAMMAR, start='value')))
def test_generates_valid_json(string):
    if False:
        while True:
            i = 10
    json.loads(string)

@pytest.mark.parametrize('start, type_', [('dict', dict), ('list', list), ('STRING', str), ('NUMBER', (int, float)), ('TRUE', bool), ('FALSE', bool), ('NULL', type(None))])
@given(data=data())
def test_can_specify_start_rule(data, start, type_):
    if False:
        while True:
            i = 10
    string = data.draw(from_lark(Lark(EBNF_GRAMMAR, start='value'), start=start))
    value = json.loads(string)
    assert isinstance(value, type_)

def test_can_generate_ignored_tokens():
    if False:
        for i in range(10):
            print('nop')
    list_grammar = '\n    list : "[" [STRING ("," STRING)*] "]"\n    STRING : /"[a-z]*"/\n    WS : /[ \\t\\r\\n]+/\n    %ignore WS\n    '
    strategy = from_lark(Lark(list_grammar, start='list'))
    find_any(strategy, lambda s: '\t' in s)

def test_generation_without_whitespace():
    if False:
        print('Hello World!')
    find_any(from_lark(Lark(LIST_GRAMMAR, start='list')), lambda g: ' ' not in g)

def test_cannot_convert_EBNF_to_strategy_directly():
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidArgument):
        from_lark(EBNF_GRAMMAR).example()
    with pytest.raises(TypeError):
        from_lark(EBNF_GRAMMAR, start='value').example()
    with pytest.raises(InvalidArgument):
        from_lark(Lark(LIST_GRAMMAR, start='list'), explicit=[]).example()

def test_undefined_terminals_require_explicit_strategies():
    if False:
        i = 10
        return i + 15
    elem_grammar = '\n    list : "[" [ELEMENT ("," ELEMENT)*] "]"\n    %declare ELEMENT\n    '
    with pytest.raises(InvalidArgument):
        from_lark(Lark(elem_grammar, start='list')).example()
    strategy = {'ELEMENT': just('200')}
    from_lark(Lark(elem_grammar, start='list'), explicit=strategy).example()

def test_cannot_use_explicit_strategies_for_unknown_terminals():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        from_lark(Lark(LIST_GRAMMAR, start='list'), explicit={'unused_name': just('')}).example()

def test_non_string_explicit_strategies_are_invalid():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidArgument):
        from_lark(Lark(LIST_GRAMMAR, start='list'), explicit={'NUMBER': just(0)}).example()

@given(string=from_lark(Lark(LIST_GRAMMAR, start='list'), explicit={'NUMBER': just('0')}))
def test_can_override_defined_terminal(string):
    if False:
        return 10
    assert sum(json.loads(string)) == 0
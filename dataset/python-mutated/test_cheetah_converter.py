""""""
import functools
import grc.converter.cheetah_converter as parser

def test_basic():
    if False:
        print('Hello World!')
    c = parser.Converter(names={'abc'})
    for convert in (c.convert_simple, c.convert_hard, c.to_python):
        assert 'abc' == convert('$abc')
        assert 'abc' == convert('$abc()')
        assert 'abc' == convert('$(abc)')
        assert 'abc' == convert('$(abc())')
        assert 'abc' == convert('${abc}')
        assert 'abc' == convert('${abc()}')
    assert c.stats['simple'] == 2 * 6
    assert c.stats['hard'] == 1 * 6

def test_simple():
    if False:
        for i in range(10):
            print('nop')
    convert = parser.Converter(names={'abc': {'def'}})
    assert 'abc' == convert.convert_simple('$abc')
    assert 'abc.def' == convert.convert_simple('$abc.def')
    assert 'abc.def' == convert.convert_simple('$(abc.def)')
    assert 'abc.def' == convert.convert_simple('${abc.def}')
    try:
        convert.convert_simple('$abc.not_a_sub_key')
    except NameError:
        assert True
    else:
        assert False

def test_conditional():
    if False:
        i = 10
        return i + 15
    convert = parser.Converter(names={'abc'})
    assert '(asb_asd_ if abc > 0 else __not__)' == convert.convert_inline_conditional('#if $abc > 0 then asb_$asd_ else __not__')

def test_simple_format_string():
    if False:
        for i in range(10):
            print('nop')
    convert = functools.partial(parser.Converter(names={'abc'}).convert_simple, spec=parser.FormatString)
    assert '{abc}' == convert('$abc')
    assert '{abc:eval}' == convert('$abc()')
    assert '{abc}' == convert('$(abc)')
    assert '{abc:eval}' == convert('$(abc())')
    assert '{abc}' == convert('${abc}')
    assert '{abc:eval}' == convert('${abc()}')

def test_hard_format_string():
    if False:
        for i in range(10):
            print('nop')
    names = {'abc': {'ff'}, 'param1': {}, 'param2': {}}
    convert = functools.partial(parser.Converter(names).convert_hard, spec=parser.FormatString)
    assert 'make_a_cool_block_{abc.ff}({param1}, {param2})' == convert('make_a_cool_block_${abc.ff}($param1, $param2)')
converter = parser.Converter(names={'abc'})
c2p = converter.to_python

def test_opts():
    if False:
        i = 10
        return i + 15
    assert 'abc abc abc' == c2p('$abc $(abc) ${abc}')
    assert 'abc abc.abc abc' == c2p('$abc $abc.abc ${abc}')
    assert 'abc abc[].abc abc' == c2p('$abc $abc[].abc() ${abc}')

def test_nested():
    if False:
        for i in range(10):
            print('nop')
    assert 'abc(abc) abc + abc abc[abc]' == c2p('$abc($abc) $(abc + $abc) ${abc[$abc]}')
    assert '(abc_abc_)' == c2p('(abc_$(abc)_)')

def test_nested2():
    if False:
        for i in range(10):
            print('nop')

    class Other(parser.Python):
        nested_start = '{'
        nested_end = '}'
    assert 'abc({abc})' == converter.convert('$abc($abc)', spec=Other)

def test_nested3():
    if False:
        for i in range(10):
            print('nop')

    class Other(parser.Python):
        start = '{'
        end = '}'
    assert '{abc(abc)}' == converter.convert('$abc($abc)', spec=Other)

def test_with_string():
    if False:
        for i in range(10):
            print('nop')
    assert 'abc "$(abc)" abc' == c2p('$abc "$(abc)" ${abc}')
    assert "abc '$(abc)' abc" == c2p("$abc '$(abc)' ${abc}")
    assert 'abc "\'\'$(abc)" abc' == c2p('$abc "\'\'$(abc)" ${abc}')

def test_if():
    if False:
        i = 10
        return i + 15
    result = converter.to_mako('\n        #if $abc > 0\n            test\n        #else if $abc < 0\n            test\n        #else\n            bla\n        #end if\n    ')
    expected = '\n        % if abc > 0:\n            test\n        % elif abc < 0:\n            test\n        % else:\n            bla\n        % endif\n    '
    assert result == expected

def test_hash_end():
    if False:
        return 10
    result = converter.to_mako('$abc#slurp')
    assert result == '${abc}\\'

def test_slurp_if():
    if False:
        return 10
    result = converter.to_mako('\n        $abc#slurp\n        #if $abc\n    ')
    expected = '\n        ${abc}\n        % if abc:\n    '
    assert result == expected
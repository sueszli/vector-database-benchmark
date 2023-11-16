import pyparsing
import pytest
from hypothesis import example
from hypothesis import given
from hypothesis.strategies import text
from mitmproxy import command_lexer

@pytest.mark.parametrize('test_input,valid', [("'foo'", True), ('"foo"', True), ("'foo' bar'", False), ("'foo' 'bar'", False), ("'foo'x", False), ('"foo    ', True), ('"foo \'bar\'   ', True), ('"foo\\', True)])
def test_partial_quoted_string(test_input, valid):
    if False:
        for i in range(10):
            print('nop')
    if valid:
        assert command_lexer.PartialQuotedString.parseString(test_input, parseAll=True)[0] == test_input
    else:
        with pytest.raises(pyparsing.ParseException):
            command_lexer.PartialQuotedString.parseString(test_input, parseAll=True)

@pytest.mark.parametrize('test_input,expected', [("'foo'", ["'foo'"]), ('"foo"', ['"foo"']), ("'foo' 'bar'", ["'foo'", ' ', "'bar'"]), ("'foo'x", ["'foo'", 'x']), ('"foo', ['"foo']), ('"foo \'bar\' ', ['"foo \'bar\' ']), ('"foo\\', ['"foo\\'])])
def test_expr(test_input, expected):
    if False:
        for i in range(10):
            print('nop')
    assert list(command_lexer.expr.parseString(test_input, parseAll=True)) == expected

@given(text())
@example('foo')
@example("'foo\\''")
@example('\'foo\\"\'')
@example('"foo\\""')
@example('"foo\\\'"')
@example("'foo\\'")
@example("'foo\\\\'")
@example('"foo\\\'"')
@example('"foo\\\\\'"')
@example('\'foo\\"\'')
@example('\\\\\\foo')
def test_quote_unquote_cycle(s):
    if False:
        for i in range(10):
            print('nop')
    assert command_lexer.unquote(command_lexer.quote(s)).replace('\\x22', '"') == s

@given(text())
@example("'foo\\'")
def test_unquote_never_fails(s):
    if False:
        return 10
    command_lexer.unquote(s)
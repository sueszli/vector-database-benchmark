"""Tests for tokenutil"""
import pytest
from IPython.utils.tokenutil import token_at_cursor, line_at_cursor

def expect_token(expected, cell, cursor_pos):
    if False:
        print('Hello World!')
    token = token_at_cursor(cell, cursor_pos)
    offset = 0
    for line in cell.splitlines():
        if offset + len(line) >= cursor_pos:
            break
        else:
            offset += len(line) + 1
    column = cursor_pos - offset
    line_with_cursor = '%s|%s' % (line[:column], line[column:])
    assert token == expected, 'Expected %r, got %r in: %r (pos %i)' % (expected, token, line_with_cursor, cursor_pos)

def test_simple():
    if False:
        i = 10
        return i + 15
    cell = 'foo'
    for i in range(len(cell)):
        expect_token('foo', cell, i)

def test_function():
    if False:
        for i in range(10):
            print('nop')
    cell = "foo(a=5, b='10')"
    expected = 'foo'
    for i in range(cell.find('a=') + 1):
        expect_token('foo', cell, i)
    for i in [cell.find('=') + 1, cell.rfind('=') + 1]:
        expect_token('foo', cell, i)
    for i in range(cell.find(','), cell.find('b=')):
        expect_token('foo', cell, i)

def test_multiline():
    if False:
        print('Hello World!')
    cell = '\n'.join(['a = 5', 'b = hello("string", there)'])
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)

def test_multiline_token():
    if False:
        print('Hello World!')
    cell = '\n'.join(['"""\n\nxxxxxxxxxx\n\n"""', '5, """', 'docstring', 'multiline token', '""", [', '2, 3, "complicated"]', 'b = hello("string", there)'])
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)

def test_nested_call():
    if False:
        i = 10
        return i + 15
    cell = 'foo(bar(a=5), b=10)'
    expected = 'foo'
    start = cell.index('bar') + 1
    for i in range(start, start + 3):
        expect_token(expected, cell, i)
    expected = 'bar'
    start = cell.index('a=')
    for i in range(start, start + 3):
        expect_token(expected, cell, i)
    expected = 'foo'
    start = cell.index(')') + 1
    for i in range(start, len(cell) - 1):
        expect_token(expected, cell, i)

def test_attrs():
    if False:
        while True:
            i = 10
    cell = 'a = obj.attr.subattr'
    expected = 'obj'
    idx = cell.find('obj') + 1
    for i in range(idx, idx + 3):
        expect_token(expected, cell, i)
    idx = cell.find('.attr') + 2
    expected = 'obj.attr'
    for i in range(idx, idx + 4):
        expect_token(expected, cell, i)
    idx = cell.find('.subattr') + 2
    expected = 'obj.attr.subattr'
    for i in range(idx, len(cell)):
        expect_token(expected, cell, i)

def test_line_at_cursor():
    if False:
        return 10
    cell = ''
    (line, offset) = line_at_cursor(cell, cursor_pos=11)
    assert line == ''
    assert offset == 0
    cell = 'One\nTwo\n'
    (line, offset) = line_at_cursor(cell, cursor_pos=4)
    assert line == 'Two\n'
    assert offset == 4
    cell = 'pri\npri'
    (line, offset) = line_at_cursor(cell, cursor_pos=7)
    assert line == 'pri'
    assert offset == 4

@pytest.mark.parametrize('c, token', zip(list(range(16, 22)) + list(range(22, 28)), ['int'] * (22 - 16) + ['map'] * (28 - 22)))
def test_multiline_statement(c, token):
    if False:
        while True:
            i = 10
    cell = 'a = (1,\n    3)\n\nint()\nmap()\n'
    expect_token(token, cell, c)
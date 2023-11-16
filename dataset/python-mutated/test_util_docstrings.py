"""Test sphinx.util.docstrings."""
from sphinx.util.docstrings import prepare_commentdoc, prepare_docstring, separate_metadata

def test_separate_metadata():
    if False:
        return 10
    text = ':meta foo: bar\n:meta baz:\n'
    (docstring, metadata) = separate_metadata(text)
    assert docstring == ''
    assert metadata == {'foo': 'bar', 'baz': ''}
    text = ':meta foo: bar\n:param baz:\n'
    (docstring, metadata) = separate_metadata(text)
    assert docstring == ':param baz:\n'
    assert metadata == {'foo': 'bar'}
    text = 'blah blah blah\n:meta foo: bar\n:meta baz:\n'
    (docstring, metadata) = separate_metadata(text)
    assert docstring == text
    assert metadata == {}
    text = 'blah blah blah\n\n:meta foo: bar\n:meta baz:\n'
    (docstring, metadata) = separate_metadata(text)
    assert docstring == 'blah blah blah\n\n'
    assert metadata == {'foo': 'bar', 'baz': ''}
    text = ':meta foo: bar\nblah blah blah\n:meta baz:\n'
    (docstring, metadata) = separate_metadata(text)
    assert docstring == 'blah blah blah\n:meta baz:\n'
    assert metadata == {'foo': 'bar'}

def test_prepare_docstring():
    if False:
        for i in range(10):
            print('nop')
    docstring = 'multiline docstring\n\n                Lorem ipsum dolor sit amet, consectetur adipiscing elit,\n                sed do eiusmod tempor incididunt ut labore et dolore magna\n                aliqua::\n\n                  Ut enim ad minim veniam, quis nostrud exercitation\n                    ullamco laboris nisi ut aliquip ex ea commodo consequat.\n                '
    assert prepare_docstring(docstring) == ['multiline docstring', '', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit,', 'sed do eiusmod tempor incididunt ut labore et dolore magna', 'aliqua::', '', '  Ut enim ad minim veniam, quis nostrud exercitation', '    ullamco laboris nisi ut aliquip ex ea commodo consequat.', '']
    docstring = '\n\n                multiline docstring with leading empty lines\n                '
    assert prepare_docstring(docstring) == ['multiline docstring with leading empty lines', '']
    docstring = 'single line docstring'
    assert prepare_docstring(docstring) == ['single line docstring', '']

def test_prepare_commentdoc():
    if False:
        for i in range(10):
            print('nop')
    assert prepare_commentdoc('hello world') == []
    assert prepare_commentdoc('#: hello world') == ['hello world', '']
    assert prepare_commentdoc('#:  hello world') == [' hello world', '']
    assert prepare_commentdoc('#: hello\n#: world\n') == ['hello', 'world', '']
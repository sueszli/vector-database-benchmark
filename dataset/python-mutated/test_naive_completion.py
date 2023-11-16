import pytest
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document

@pytest.fixture
def completer():
    if False:
        for i in range(10):
            print('nop')
    import mycli.sqlcompleter as sqlcompleter
    return sqlcompleter.SQLCompleter(smart_completion=False)

@pytest.fixture
def complete_event():
    if False:
        return 10
    from unittest.mock import Mock
    return Mock()

def test_empty_string_completion(completer, complete_event):
    if False:
        print('Hello World!')
    text = ''
    position = 0
    result = list(completer.get_completions(Document(text=text, cursor_position=position), complete_event))
    assert result == list(map(Completion, completer.all_completions))

def test_select_keyword_completion(completer, complete_event):
    if False:
        return 10
    text = 'SEL'
    position = len('SEL')
    result = list(completer.get_completions(Document(text=text, cursor_position=position), complete_event))
    assert result == list([Completion(text='SELECT', start_position=-3)])

def test_function_name_completion(completer, complete_event):
    if False:
        print('Hello World!')
    text = 'SELECT MA'
    position = len('SELECT MA')
    result = list(completer.get_completions(Document(text=text, cursor_position=position), complete_event))
    assert sorted((x.text for x in result)) == ['MASTER', 'MAX']

def test_column_name_completion(completer, complete_event):
    if False:
        i = 10
        return i + 15
    text = 'SELECT  FROM users'
    position = len('SELECT ')
    result = list(completer.get_completions(Document(text=text, cursor_position=position), complete_event))
    assert result == list(map(Completion, completer.all_completions))

def test_special_name_completion(completer, complete_event):
    if False:
        print('Hello World!')
    text = '\\'
    position = len('\\')
    result = set(completer.get_completions(Document(text=text, cursor_position=position), complete_event))
    assert result == set()
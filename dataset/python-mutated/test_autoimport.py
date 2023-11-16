from typing import Any, Dict, List
from unittest.mock import Mock
import jedi
import parso
import pytest
from pylsp import lsp, uris
from pylsp.config.config import Config
from pylsp.plugins.rope_autoimport import _get_score, _should_insert, get_name_or_module, get_names
from pylsp.plugins.rope_autoimport import pylsp_completions as pylsp_autoimport_completions
from pylsp.plugins.rope_autoimport import pylsp_initialize
from pylsp.workspace import Workspace
DOC_URI = uris.from_fs_path(__file__)

def contains_autoimport(suggestion: Dict[str, Any], module: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks if `suggestion` contains an autoimport for `module`.'
    return suggestion.get('label', '') == module and 'import' in suggestion.get('detail', '')

@pytest.fixture(scope='session')
def autoimport_workspace(tmp_path_factory) -> Workspace:
    if False:
        while True:
            i = 10
    'Special autoimport workspace. Persists across sessions to make in-memory sqlite3 database fast.'
    workspace = Workspace(uris.from_fs_path(str(tmp_path_factory.mktemp('pylsp'))), Mock())
    workspace._config = Config(workspace.root_uri, {}, 0, {})
    workspace._config.update({'rope_autoimport': {'memory': True, 'enabled': True, 'completions': {'enabled': True}, 'code_actions': {'enabled': True}}})
    pylsp_initialize(workspace._config, workspace)
    yield workspace
    workspace.close()

@pytest.fixture
def completions(config: Config, autoimport_workspace: Workspace, request):
    if False:
        i = 10
        return i + 15
    (document, position) = request.param
    com_position = {'line': 0, 'character': position}
    autoimport_workspace.put_document(DOC_URI, source=document)
    doc = autoimport_workspace.get_document(DOC_URI)
    yield pylsp_autoimport_completions(config, autoimport_workspace, doc, com_position, None)
    autoimport_workspace.rm_document(DOC_URI)

def should_insert(phrase: str, position: int):
    if False:
        while True:
            i = 10
    expr = parso.parse(phrase)
    word_node = expr.get_leaf_for_position((1, position))
    return _should_insert(expr, word_node)

def check_dict(query: Dict, results: List[Dict]) -> bool:
    if False:
        while True:
            i = 10
    for result in results:
        if all((result[key] == query[key] for key in query.keys())):
            return True
    return False

@pytest.mark.parametrize('completions', [('pathli ', 6)], indirect=True)
def test_autoimport_completion(completions):
    if False:
        return 10
    assert completions
    assert check_dict({'label': 'pathlib', 'kind': lsp.CompletionItemKind.Module}, completions)

@pytest.mark.parametrize('completions', [('import ', 7)], indirect=True)
def test_autoimport_import(completions):
    if False:
        for i in range(10):
            print('nop')
    assert len(completions) == 0

@pytest.mark.parametrize('completions', [('pathlib', 2)], indirect=True)
def test_autoimport_pathlib(completions):
    if False:
        for i in range(10):
            print('nop')
    assert completions[0]['label'] == 'pathlib'
    start = {'line': 0, 'character': 0}
    edit_range = {'start': start, 'end': start}
    assert completions[0]['additionalTextEdits'] == [{'range': edit_range, 'newText': 'import pathlib\n'}]

@pytest.mark.parametrize('completions', [('import test\n', 10)], indirect=True)
def test_autoimport_import_with_name(completions):
    if False:
        print('Hello World!')
    assert len(completions) == 0

@pytest.mark.parametrize('completions', [('def func(s', 10)], indirect=True)
def test_autoimport_function(completions):
    if False:
        print('Hello World!')
    assert len(completions) == 0

@pytest.mark.parametrize('completions', [('class Test', 10)], indirect=True)
def test_autoimport_class(completions):
    if False:
        i = 10
        return i + 15
    assert len(completions) == 0

@pytest.mark.parametrize('completions', [('\n', 0)], indirect=True)
def test_autoimport_empty_line(completions):
    if False:
        return 10
    assert len(completions) == 0

@pytest.mark.parametrize('completions', [('class Test(NamedTupl):', 20)], indirect=True)
def test_autoimport_class_complete(completions):
    if False:
        while True:
            i = 10
    assert len(completions) > 0

@pytest.mark.parametrize('completions', [('class Test(NamedTupl', 20)], indirect=True)
def test_autoimport_class_incomplete(completions):
    if False:
        return 10
    assert len(completions) > 0

@pytest.mark.parametrize('completions', [('def func(s:Lis', 12)], indirect=True)
def test_autoimport_function_typing(completions):
    if False:
        while True:
            i = 10
    assert len(completions) > 0
    assert check_dict({'label': 'List'}, completions)

@pytest.mark.parametrize('completions', [('def func(s : Lis ):', 16)], indirect=True)
def test_autoimport_function_typing_complete(completions):
    if False:
        return 10
    assert len(completions) > 0
    assert check_dict({'label': 'List'}, completions)

@pytest.mark.parametrize('completions', [('def func(s : Lis ) -> Generat:', 29)], indirect=True)
def test_autoimport_function_typing_return(completions):
    if False:
        while True:
            i = 10
    assert len(completions) > 0
    assert check_dict({'label': 'Generator'}, completions)

def test_autoimport_defined_name(config, workspace):
    if False:
        for i in range(10):
            print('nop')
    document = 'List = "hi"\nLis'
    com_position = {'line': 1, 'character': 3}
    workspace.put_document(DOC_URI, source=document)
    doc = workspace.get_document(DOC_URI)
    completions = pylsp_autoimport_completions(config, workspace, doc, com_position, None)
    workspace.rm_document(DOC_URI)
    assert not check_dict({'label': 'List'}, completions)

class TestShouldInsert:

    def test_dot(self):
        if False:
            return 10
        assert not should_insert('str.', 4)

    def test_dot_partial(self):
        if False:
            while True:
                i = 10
        assert not should_insert('str.metho\n', 9)

    def test_comment(self):
        if False:
            i = 10
            return i + 15
        assert not should_insert('#', 1)

    def test_comment_indent(self):
        if False:
            return 10
        assert not should_insert('    # ', 5)

    def test_from(self):
        if False:
            i = 10
            return i + 15
        assert not should_insert('from ', 5)
        assert should_insert('from ', 4)

def test_sort_sources():
    if False:
        print('Hello World!')
    result1 = _get_score(1, 'import pathlib', 'pathlib', 'pathli')
    result2 = _get_score(2, 'import pathlib', 'pathlib', 'pathli')
    assert result1 < result2

def test_sort_statements():
    if False:
        return 10
    result1 = _get_score(2, 'from importlib_metadata import pathlib', 'pathlib', 'pathli')
    result2 = _get_score(2, 'import pathlib', 'pathlib', 'pathli')
    assert result1 > result2

def test_sort_both():
    if False:
        while True:
            i = 10
    result1 = _get_score(3, 'from importlib_metadata import pathlib', 'pathlib', 'pathli')
    result2 = _get_score(2, 'import pathlib', 'pathlib', 'pathli')
    assert result1 > result2

def test_get_names():
    if False:
        print('Hello World!')
    source = '\n    from a import s as e\n    import blah, bleh\n    hello = "str"\n    a, b = 1, 2\n    def someone():\n        soemthing\n    class sfa:\n        sfiosifo\n    '
    results = get_names(jedi.Script(code=source))
    assert results == set(['blah', 'bleh', 'e', 'hello', 'someone', 'sfa', 'a', 'b'])

@pytest.mark.parametrize('message', ['Undefined name `os`', "F821 undefined name 'numpy'", "undefined name 'numpy'"])
def test_autoimport_code_actions_get_correct_module_name(autoimport_workspace, message):
    if False:
        return 10
    source = "os.path.join('a', 'b')"
    autoimport_workspace.put_document(DOC_URI, source=source)
    doc = autoimport_workspace.get_document(DOC_URI)
    diagnostic = {'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 2}}, 'message': message}
    module_name = get_name_or_module(doc, diagnostic)
    autoimport_workspace.rm_document(DOC_URI)
    assert module_name == 'os'
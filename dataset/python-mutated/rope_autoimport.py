import logging
from typing import Any, Dict, Generator, List, Optional, Set, Union
import parso
from jedi import Script
from parso.python import tree
from parso.tree import NodeOrLeaf
from rope.base.resources import Resource
from rope.contrib.autoimport.defs import SearchResult
from rope.contrib.autoimport.sqlite import AutoImport
from pylsp import hookimpl
from pylsp.config.config import Config
from pylsp.workspace import Document, Workspace
from ._rope_task_handle import PylspTaskHandle
log = logging.getLogger(__name__)
_score_pow = 5
_score_max = 10 ** _score_pow
MAX_RESULTS_COMPLETIONS = 1000
MAX_RESULTS_CODE_ACTIONS = 5

@hookimpl
def pylsp_settings() -> Dict[str, Dict[str, Dict[str, Any]]]:
    if False:
        while True:
            i = 10
    return {'plugins': {'rope_autoimport': {'enabled': False, 'memory': False, 'completions': {'enabled': True}, 'code_actions': {'enabled': True}}}}

def _should_insert(expr: tree.BaseNode, word_node: tree.Leaf) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if we should insert the word_node on the given expr.\n\n    Works for both correct and incorrect code. This is because the\n    user is often working on the code as they write it.\n    '
    if not word_node:
        return False
    if len(expr.children) == 0:
        return True
    first_child = expr.children[0]
    if isinstance(first_child, tree.EndMarker):
        if '#' in first_child.prefix:
            return False
    if first_child == word_node:
        return True
    if len(expr.children) > 1:
        if any((node.type == 'operator' and '.' in node.value or node.type == 'trailer' for node in expr.children)):
            return False
    if isinstance(first_child, (tree.PythonErrorNode, tree.PythonNode)):
        return _should_insert(first_child, word_node)
    return _handle_first_child(first_child, expr, word_node)

def _handle_first_child(first_child: NodeOrLeaf, expr: tree.BaseNode, word_node: tree.Leaf) -> bool:
    if False:
        print('Hello World!')
    'Check if we suggest imports given the following first child.'
    if isinstance(first_child, tree.Import):
        return False
    if isinstance(first_child, (tree.PythonLeaf, tree.PythonErrorLeaf)):
        if first_child.value in ('import', 'from'):
            return False
    if isinstance(first_child, tree.Keyword):
        if first_child.value == 'def':
            return _should_import_function(word_node, expr)
        if first_child.value == 'class':
            return _should_import_class(word_node, expr)
    return True

def _should_import_class(word_node: tree.Leaf, expr: tree.BaseNode) -> bool:
    if False:
        i = 10
        return i + 15
    prev_node = None
    for node in expr.children:
        if isinstance(node, tree.Name):
            if isinstance(prev_node, tree.Operator):
                if node == word_node and prev_node.value == '(':
                    return True
        prev_node = node
    return False

def _should_import_function(word_node: tree.Leaf, expr: tree.BaseNode) -> bool:
    if False:
        while True:
            i = 10
    prev_node = None
    for node in expr.children:
        if _handle_argument(node, word_node):
            return True
        if isinstance(prev_node, tree.Operator):
            if prev_node.value == '->':
                if node == word_node:
                    return True
        prev_node = node
    return False

def _handle_argument(node: NodeOrLeaf, word_node: tree.Leaf):
    if False:
        print('Hello World!')
    if isinstance(node, tree.PythonNode):
        if node.type == 'tfpdef':
            if node.children[2] == word_node:
                return True
        if node.type == 'parameters':
            for parameter in node.children:
                if _handle_argument(parameter, word_node):
                    return True
    return False

def _process_statements(suggestions: List[SearchResult], doc_uri: str, word: str, autoimport: AutoImport, document: Document, feature: str='completions') -> Generator[Dict[str, Any], None, None]:
    if False:
        i = 10
        return i + 15
    for suggestion in suggestions:
        insert_line = autoimport.find_insertion_line(document.source) - 1
        start = {'line': insert_line, 'character': 0}
        edit_range = {'start': start, 'end': start}
        edit = {'range': edit_range, 'newText': suggestion.import_statement + '\n'}
        score = _get_score(suggestion.source, suggestion.import_statement, suggestion.name, word)
        if score > _score_max:
            continue
        if feature == 'completions':
            yield {'label': suggestion.name, 'kind': suggestion.itemkind, 'sortText': _sort_import(score), 'data': {'doc_uri': doc_uri}, 'detail': _document(suggestion.import_statement), 'additionalTextEdits': [edit]}
        elif feature == 'code_actions':
            yield {'title': suggestion.import_statement, 'kind': 'quickfix', 'edit': {'changes': {doc_uri: [edit]}}, 'data': {'sortText': _sort_import(score)}}
        else:
            raise ValueError(f'Unknown feature: {feature}')

def get_names(script: Script) -> Set[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get all names to ignore from the current file.'
    raw_names = script.get_names(definitions=True)
    log.debug(raw_names)
    return set((name.name for name in raw_names))

@hookimpl
def pylsp_completions(config: Config, workspace: Workspace, document: Document, position, ignored_names: Union[Set[str], None]):
    if False:
        while True:
            i = 10
    'Get autoimport suggestions.'
    if not config.plugin_settings('rope_autoimport').get('completions', {}).get('enabled', True):
        return []
    line = document.lines[position['line']]
    expr = parso.parse(line)
    word_node = expr.get_leaf_for_position((1, position['character']))
    if not _should_insert(expr, word_node):
        return []
    word = word_node.value
    log.debug(f'autoimport: searching for word: {word}')
    rope_config = config.settings(document_path=document.path).get('rope', {})
    ignored_names: Set[str] = ignored_names or get_names(document.jedi_script(use_document_path=True))
    autoimport = workspace._rope_autoimport(rope_config)
    suggestions = list(autoimport.search_full(word, ignored_names=ignored_names))
    results = list(sorted(_process_statements(suggestions, document.uri, word, autoimport, document, 'completions'), key=lambda statement: statement['sortText']))
    if len(results) > MAX_RESULTS_COMPLETIONS:
        results = results[:MAX_RESULTS_COMPLETIONS]
    return results

def _document(import_statement: str) -> str:
    if False:
        print('Hello World!')
    return '# Auto-Import\n' + import_statement

def _get_score(source: int, full_statement: str, suggested_name: str, desired_name) -> int:
    if False:
        while True:
            i = 10
    import_length = len('import')
    full_statement_score = len(full_statement) - import_length
    suggested_name_score = (len(suggested_name) - len(desired_name)) ** 2
    source_score = 20 * source
    return suggested_name_score + full_statement_score + source_score

def _sort_import(score: int) -> str:
    if False:
        while True:
            i = 10
    score = max(min(score, _score_max - 1), 0)
    return '[z' + str(score).rjust(_score_pow, '0')

def get_name_or_module(document, diagnostic) -> str:
    if False:
        return 10
    start = diagnostic['range']['start']
    return parso.parse(document.lines[start['line']]).get_leaf_for_position((1, start['character'] + 1)).value

@hookimpl
def pylsp_code_actions(config: Config, workspace: Workspace, document: Document, range: Dict, context: Dict) -> List[Dict]:
    if False:
        while True:
            i = 10
    '\n    Provide code actions through rope.\n\n    Parameters\n    ----------\n    config : pylsp.config.config.Config\n        Current config.\n    workspace : pylsp.workspace.Workspace\n        Current workspace.\n    document : pylsp.workspace.Document\n        Document to apply code actions on.\n    range : Dict\n        Range argument given by pylsp. Not used here.\n    context : Dict\n        CodeActionContext given as dict.\n\n    Returns\n    -------\n      List of dicts containing the code actions.\n    '
    if not config.plugin_settings('rope_autoimport').get('code_actions', {}).get('enabled', True):
        return []
    log.debug(f'textDocument/codeAction: {document} {range} {context}')
    code_actions = []
    for diagnostic in context.get('diagnostics', []):
        if 'undefined name' not in diagnostic.get('message', '').lower():
            continue
        word = get_name_or_module(document, diagnostic)
        log.debug(f'autoimport: searching for word: {word}')
        rope_config = config.settings(document_path=document.path).get('rope', {})
        autoimport = workspace._rope_autoimport(rope_config, feature='code_actions')
        suggestions = list(autoimport.search_full(word))
        log.debug('autoimport: suggestions: %s', suggestions)
        results = list(sorted(_process_statements(suggestions, document.uri, word, autoimport, document, 'code_actions'), key=lambda statement: statement['data']['sortText']))
        if len(results) > MAX_RESULTS_CODE_ACTIONS:
            results = results[:MAX_RESULTS_CODE_ACTIONS]
        code_actions.extend(results)
    return code_actions

def _reload_cache(config: Config, workspace: Workspace, files: Optional[List[Document]]=None):
    if False:
        for i in range(10):
            print('nop')
    memory: bool = config.plugin_settings('rope_autoimport').get('memory', False)
    rope_config = config.settings().get('rope', {})
    autoimport = workspace._rope_autoimport(rope_config, memory)
    task_handle = PylspTaskHandle(workspace)
    resources: Optional[List[Resource]] = None if files is None else [document._rope_resource(rope_config) for document in files]
    autoimport.generate_cache(task_handle=task_handle, resources=resources)
    autoimport.generate_modules_cache(task_handle=task_handle)

@hookimpl
def pylsp_initialize(config: Config, workspace: Workspace):
    if False:
        while True:
            i = 10
    'Initialize AutoImport.\n\n    Generates the cache for local and global items.\n    '
    _reload_cache(config, workspace)

@hookimpl
def pylsp_document_did_open(config: Config, workspace: Workspace):
    if False:
        return 10
    'Initialize AutoImport.\n\n    Generates the cache for local and global items.\n    '
    _reload_cache(config, workspace)

@hookimpl
def pylsp_document_did_save(config: Config, workspace: Workspace, document: Document):
    if False:
        while True:
            i = 10
    'Update the names associated with this document.'
    _reload_cache(config, workspace, [document])

@hookimpl
def pylsp_workspace_configuration_changed(config: Config, workspace: Workspace):
    if False:
        return 10
    '\n    Initialize autoimport if it has been enabled through a\n    workspace/didChangeConfiguration message from the frontend.\n\n    Generates the cache for local and global items.\n    '
    if config.plugin_settings('rope_autoimport').get('enabled', False):
        _reload_cache(config, workspace)
    else:
        log.debug('autoimport: Skipping cache reload.')
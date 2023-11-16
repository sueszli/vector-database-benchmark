from __future__ import annotations
import ast
import sys
from pathlib import Path
import click
from reactpy import html
from reactpy._console.ast_utils import ChangedNode, find_element_constructor_usages, rewrite_changed_nodes

@click.command()
@click.argument('paths', nargs=-1, type=click.Path(exists=True))
def rewrite_keys(paths: list[str]) -> None:
    if False:
        print('Hello World!')
    'Rewrite files under the given paths using the new html element API.\n\n    The old API required users to pass a dictionary of attributes to html element\n    constructor functions. For example:\n\n    >>> html.div({"className": "x"}, "y")\n    {"tagName": "div", "attributes": {"className": "x"}, "children": ["y"]}\n\n    The latest API though allows for attributes to be passed as snake_cased keyword\n    arguments instead. The above example would be rewritten as:\n\n    >>> html.div("y", class_name="x")\n    {"tagName": "div", "attributes": {"class_name": "x"}, "children": ["y"]}\n\n    All snake_case attributes are converted to camelCase by the client where necessary.\n\n    ----- Notes -----\n\n    While this command does it\'s best to preserve as much of the original code as\n    possible, there are inevitably some limitations in doing this. As a result, we\n    recommend running your code formatter like Black against your code after executing\n    this command.\n\n    Additionally, We are unable to preserve the location of comments that lie within any\n    rewritten code. This command will place the comments in the code it plans to rewrite\n    just above its changes. As such it requires manual intervention to put those\n    comments back in their original location.\n    '
    if sys.version_info < (3, 9):
        msg = 'This command requires Python>=3.9'
        raise RuntimeError(msg)
    for p in map(Path, paths):
        for f in [p] if p.is_file() else p.rglob('*.py'):
            result = generate_rewrite(file=f, source=f.read_text())
            if result is not None:
                f.write_text(result)

def generate_rewrite(file: Path, source: str) -> str | None:
    if False:
        return 10
    tree = ast.parse(source)
    changed = find_nodes_to_change(tree)
    if not changed:
        log_could_not_rewrite(file, tree)
        return None
    new = rewrite_changed_nodes(file, source, tree, changed)
    log_could_not_rewrite(file, ast.parse(new))
    return new

def find_nodes_to_change(tree: ast.AST) -> list[ChangedNode]:
    if False:
        print('Hello World!')
    changed: list[ChangedNode] = []
    for el_info in find_element_constructor_usages(tree, add_props=True):
        for kw in list(el_info.call.keywords):
            if kw.arg == 'key':
                break
        else:
            continue
        if isinstance(el_info.props, ast.Dict):
            el_info.props.keys.append(ast.Constant('key'))
            el_info.props.values.append(kw.value)
        else:
            el_info.props.keywords.append(ast.keyword(arg='key', value=kw.value))
        el_info.call.keywords.remove(kw)
        changed.append(ChangedNode(el_info.call, el_info.parents))
    return changed

def log_could_not_rewrite(file: Path, tree: ast.AST) -> None:
    if False:
        return 10
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name == 'vdom' or (hasattr(html, name) and any((kw.arg == 'key' for kw in node.keywords))):
            click.echo(f'Unable to rewrite usage at {file}:{node.lineno}')
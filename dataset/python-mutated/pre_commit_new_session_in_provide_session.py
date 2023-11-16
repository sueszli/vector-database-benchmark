from __future__ import annotations
import ast
import enum
import itertools
import pathlib
import sys
import typing

class _SessionArgument(typing.NamedTuple):
    argument: ast.arg
    default: ast.expr | None

def _get_session_arg_and_default(args: ast.arguments) -> _SessionArgument | None:
    if False:
        return 10
    arguments = reversed([*args.args, *args.kwonlyargs])
    defaults = reversed([*args.defaults, *args.kw_defaults])
    for (argument, default) in itertools.zip_longest(arguments, defaults, fillvalue=None):
        if argument is None:
            continue
        if argument.arg != 'session':
            continue
        return _SessionArgument(argument, default)
    return None

class _SessionDefault(enum.Enum):
    none = 'none'
    new_session = 'new_session'

def _is_new_session_or_none(value: ast.expr) -> _SessionDefault | None:
    if False:
        for i in range(10):
            print('nop')
    "Whether an expression is NEW_SESSION.\n\n    Old code written before the introduction of NEW_SESSION (and even some new\n    if the contributor wasn't made aware of the addition) generally uses None\n    as the default value, so we add that to the check as well.\n    "
    if isinstance(value, ast.Constant) and value.value is None:
        return _SessionDefault.none
    if isinstance(value, ast.Name) and value.id == 'NEW_SESSION':
        return _SessionDefault.new_session
    return None
_ALLOWED_DECORATOR_NAMES = ('overload', 'provide_session', 'abstractmethod')

def _is_decorated_correctly(nodes: list[ast.expr]) -> bool:
    if False:
        while True:
            i = 10
    'Whether expected decorators are provided.\n\n    Three decorators would allow NEW_SESSION usages:\n\n    * ``@provide_session``: The canonical case.\n    * ``@overload``: A typing overload and not something to actually execute.\n    * ``@abstractmethod``: This will be overridden in a subclass anyway.\n    '
    return any((isinstance(node, ast.Name) and node.id in _ALLOWED_DECORATOR_NAMES for node in nodes))

def _annotation_has_none(value: ast.expr | None) -> bool:
    if False:
        while True:
            i = 10
    if value is None:
        return False
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return _annotation_has_none(value.left) or _annotation_has_none(value.right)
    return False

def _iter_incorrect_new_session_usages(path: pathlib.Path) -> typing.Iterator[ast.FunctionDef]:
    if False:
        print('Hello World!')
    'Check NEW_SESSION usages outside functions decorated with provide_session.'
    for node in ast.walk(ast.parse(path.read_text('utf-8'), str(path))):
        if not isinstance(node, ast.FunctionDef):
            continue
        session = _get_session_arg_and_default(node.args)
        if session is None or session.default is None:
            continue
        if _is_decorated_correctly(node.decorator_list):
            continue
        default_kind = _is_new_session_or_none(session.default)
        if default_kind is None:
            continue
        if default_kind == _SessionDefault.none and _annotation_has_none(session.argument.annotation):
            continue
        yield node

def main(argv: list[str]) -> int:
    if False:
        return 10
    paths = (pathlib.Path(filename) for filename in argv[1:])
    errors = [(path, error) for path in paths for error in _iter_incorrect_new_session_usages(path)]
    if errors:
        print('Incorrect @provide_session and NEW_SESSION usages:', end='\n\n')
        for (path, error) in errors:
            print(f'{path}:{error.lineno}')
            print(f'\tdef {error.name}(...', end='\n\n')
        print("Only function decorated with @provide_session should use 'session: Session = NEW_SESSION'.")
        print('See: https://github.com/apache/airflow/blob/main/CONTRIBUTING.rst#database-session-handling')
    return len(errors)
if __name__ == '__main__':
    sys.exit(main(sys.argv))
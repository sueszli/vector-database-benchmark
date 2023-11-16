from __future__ import annotations
import ast
import sys
from typing import NamedTuple
from rich.console import Console
console = Console(color_system='standard', width=200)

class ImportTuple(NamedTuple):
    module: list[str]
    name: list[str]
    alias: str

def get_imports(path: str):
    if False:
        i = 10
        return i + 15
    with open(path) as fh:
        root = ast.parse(fh.read(), path)
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module: list[str] = node.names[0].name.split('.') if node.names else []
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.split('.')
        else:
            continue
        for n in node.names:
            yield ImportTuple(module=module, name=n.name.split('.'), alias=n.asname)
errors: list[str] = []
EXCEPTIONS = ['airflow/cli/commands/kubernetes_command.py']

def main() -> int:
    if False:
        i = 10
        return i + 15
    for path in sys.argv[1:]:
        import_count = 0
        local_error_count = 0
        for imp in get_imports(path):
            import_count += 1
            if len(imp.module) > 3:
                if imp.module[:4] == ['airflow', 'providers', 'cncf', 'kubernetes']:
                    if path not in EXCEPTIONS:
                        local_error_count += 1
                        errors.append(f"{path}: ({'.'.join(imp.module)})")
        console.print(f'[blue]{path}:[/] Import count: {import_count}, error_count {local_error_count}')
    if errors:
        console.print('[red]Some files imports from `airflow.providers.cncf.kubernetes` and they are not allowed.[/]\nOnly few k8s executors exceptions are allowed to use `airflow.providers.cncf.kubernetes`.')
        console.print('Error summary:')
        for error in errors:
            console.print(error)
        return 1
    else:
        console.print('[green]All good!')
    return 0
if __name__ == '__main__':
    sys.exit(main())
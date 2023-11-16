from __future__ import annotations
import re
import sys
from pathlib import Path
from rich.console import Console
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command [FILE] ...')
console = Console(color_system='standard', width=200)

def _check_file(file: Path) -> list:
    if False:
        for i in range(10):
            print('nop')
    content = file.read_text()
    return re.findall(' +\\:type .+?\\:', content)

def _join_with_newline(list_):
    if False:
        return 10
    return '\n'.join(list_)
if __name__ == '__main__':
    error_list = []
    for file in sys.argv[1:]:
        matches = _check_file(Path(file))
        if matches:
            error_list.append((file, matches))
    if error_list:
        error_message = '\n'.join([f'{f}: \n{_join_with_newline(m)}' for (f, m) in error_list])
        console.print(f'\n[red]Found files with types specified in docstring.\nThis is no longer needed since sphinx can now infer types from type annotations.[/]\n{error_message}\n')
        sys.exit(1)
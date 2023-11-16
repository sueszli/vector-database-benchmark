from __future__ import annotations
import os
from functools import total_ordering
from pathlib import Path
from typing import NamedTuple
from rich.console import Console
from airflow_breeze.utils.publish_docs_helpers import CONSOLE_WIDTH, prepare_code_snippet
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_PROJECT_DIR = Path(__file__).parents[5].resolve()
DOCS_DIR = ROOT_PROJECT_DIR / 'docs'
console = Console(force_terminal=True, color_system='standard', width=CONSOLE_WIDTH)

@total_ordering
class DocBuildError(NamedTuple):
    """Errors found in docs build."""
    file_path: str | None
    line_no: int | None
    message: str

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        left = (self.file_path, self.line_no, self.message)
        right = (other.file_path, other.line_no, other.message)
        return left == right

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __lt__(self, right):
        if False:
            for i in range(10):
                print('nop')
        file_path_a = self.file_path or ''
        file_path_b = right.file_path or ''
        line_no_a = self.line_no or 0
        line_no_b = right.line_no or 0
        left = (file_path_a, line_no_a, self.message)
        right = (file_path_b, line_no_b, right.message)
        return left < right

def display_errors_summary(build_errors: dict[str, list[DocBuildError]]) -> None:
    if False:
        return 10
    'Displays summary of errors'
    console.print()
    console.print('[red]' + '#' * 30 + ' Start docs build errors summary ' + '#' * 30 + '[/]')
    console.print()
    for (package_name, errors) in build_errors.items():
        if package_name:
            console.print('=' * 30 + f' [info]{package_name}[/] ' + '=' * 30)
        else:
            console.print('=' * 30, ' [info]General[/] ', '=' * 30)
        for (warning_no, error) in enumerate(sorted(errors), 1):
            console.print('-' * 30, f'[red]Error {warning_no:3}[/]', '-' * 20)
            console.print(error.message)
            console.print()
            if error.file_path and (not error.file_path.endswith('<unknown>')) and error.line_no:
                console.print(f'File path: {os.path.relpath(error.file_path, start=DOCS_DIR)} ({error.line_no})')
                console.print()
                console.print(prepare_code_snippet(error.file_path, error.line_no))
            elif error.file_path:
                console.print(f'File path: {error.file_path}')
    console.print()
    console.print('[red]' + '#' * 30 + ' End docs build errors summary ' + '#' * 30 + '[/]')
    console.print()

def parse_sphinx_warnings(warning_text: str, docs_dir: str) -> list[DocBuildError]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses warnings from Sphinx.\n\n    :param warning_text: warning to parse\n    :param docs_dir: documentation directory\n    :return: list of DocBuildErrors.\n    '
    sphinx_build_errors = []
    for sphinx_warning in warning_text.splitlines():
        if not sphinx_warning:
            continue
        warning_parts = sphinx_warning.split(':', 2)
        if len(warning_parts) == 3:
            try:
                sphinx_build_errors.append(DocBuildError(file_path=os.path.join(docs_dir, warning_parts[0]), line_no=int(warning_parts[1]), message=warning_parts[2]))
            except Exception:
                sphinx_build_errors.append(DocBuildError(file_path=None, line_no=None, message=sphinx_warning))
        else:
            sphinx_build_errors.append(DocBuildError(file_path=None, line_no=None, message=sphinx_warning))
    return sphinx_build_errors
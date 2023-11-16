"""Helpers for the scripts.linters.pre_commit_linter module.

Do not use this module anywhere else in the code base!
"""
from __future__ import annotations
import abc
import collections
import contextlib
import shutil
import sys
import tempfile
from typing import Dict, Iterator, List, Optional, TextIO
from .. import concurrent_task_utils

@contextlib.contextmanager
def redirect_stdout(new_target: TextIO) -> Iterator[TextIO]:
    if False:
        i = 10
        return i + 15
    'Redirect stdout to the new target.\n\n    Args:\n        new_target: TextIOWrapper. The new target to which stdout is redirected.\n\n    Yields:\n        TextIOWrapper. The new target.\n    '
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target

def get_duplicates_from_list_of_strings(strings: List[str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Returns a list of duplicate strings in the list of given strings.\n\n    Args:\n        strings: list(str). A list of strings.\n\n    Returns:\n        list(str). A list of duplicate string present in the given list of\n        strings.\n    '
    duplicates = []
    item_count_map: Dict[str, int] = collections.defaultdict(int)
    for string in strings:
        item_count_map[string] += 1
        if item_count_map[string] == 2:
            duplicates.append(string)
    return duplicates

@contextlib.contextmanager
def temp_dir(suffix: str='', prefix: str='', parent: Optional[str]=None) -> Iterator[str]:
    if False:
        while True:
            i = 10
    'Creates a temporary directory which is only usable in a `with` context.\n\n    Args:\n        suffix: str. Appended to the temporary directory.\n        prefix: str. Prepended to the temporary directory.\n        parent: str or None. The parent directory to place the temporary one. If\n            None, a platform-specific directory is used instead.\n\n    Yields:\n        str. The full path to the temporary directory.\n    '
    new_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=parent)
    try:
        yield new_dir
    finally:
        shutil.rmtree(new_dir)

def print_failure_message(failure_message: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Prints the given failure message in red color.\n\n    Args:\n        failure_message: str. The failure message to print.\n    '
    print('\x1b[91m' + failure_message + '\x1b[0m')

def print_success_message(success_message: str) -> None:
    if False:
        while True:
            i = 10
    'Prints the given success_message in red color.\n\n    Args:\n        success_message: str. The success message to print.\n    '
    print('\x1b[92m' + success_message + '\x1b[0m')

class BaseLinter(abc.ABC):
    """Base abstract linter manager for all linters."""

    @abc.abstractmethod
    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            for i in range(10):
                print('nop')
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
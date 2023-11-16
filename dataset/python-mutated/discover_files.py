"""Functions related to discovering paths."""
from __future__ import annotations
import logging
import os.path
from typing import Callable
from typing import Generator
from typing import Sequence
from flake8 import utils
LOG = logging.getLogger(__name__)

def _filenames_from(arg: str, *, predicate: Callable[[str], bool]) -> Generator[str, None, None]:
    if False:
        return 10
    'Generate filenames from an argument.\n\n    :param arg:\n        Parameter from the command-line.\n    :param predicate:\n        Predicate to use to filter out filenames. If the predicate\n        returns ``True`` we will exclude the filename, otherwise we\n        will yield it. By default, we include every filename\n        generated.\n    :returns:\n        Generator of paths\n    '
    if predicate(arg):
        return
    if os.path.isdir(arg):
        for (root, sub_directories, files) in os.walk(arg):
            for directory in tuple(sub_directories):
                joined = os.path.join(root, directory)
                if predicate(joined):
                    sub_directories.remove(directory)
            for filename in files:
                joined = os.path.join(root, filename)
                if not predicate(joined):
                    yield joined
    else:
        yield arg

def expand_paths(*, paths: Sequence[str], stdin_display_name: str, filename_patterns: Sequence[str], exclude: Sequence[str]) -> Generator[str, None, None]:
    if False:
        i = 10
        return i + 15
    'Expand out ``paths`` from commandline to the lintable files.'
    if not paths:
        paths = ['.']

    def is_excluded(arg: str) -> bool:
        if False:
            print('Hello World!')
        if arg == '-':
            if stdin_display_name == 'stdin':
                return False
            arg = stdin_display_name
        return utils.matches_filename(arg, patterns=exclude, log_message='"%(path)s" has %(whether)sbeen excluded', logger=LOG)
    return (filename for path in paths for filename in _filenames_from(path, predicate=is_excluded) if filename == '-' or path == filename or utils.fnmatch(filename, filename_patterns))
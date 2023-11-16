"""Tests for circular imports in all local packages and modules.

This ensures all internal packages can be imported right away without
any need to import some other module before doing so.

This module is based on an idea that pytest uses for self-testing:
* https://github.com/sanitizers/octomachinery/blob/be18b54/tests/circular_imports_test.py
* https://github.com/pytest-dev/pytest/blob/d18c75b/testing/test_meta.py
* https://twitter.com/codewithanthony/status/1229445110510735361
"""
import os
import pkgutil
import socket
import subprocess
import sys
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Generator, List, Union
import pytest
if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet
import aiohttp

def _mark_aiohttp_worker_for_skipping(importables: List[str]) -> List[Union[str, 'ParameterSet']]:
    if False:
        i = 10
        return i + 15
    return [pytest.param(importable, marks=pytest.mark.skipif(not hasattr(socket, 'AF_UNIX'), reason="It's a UNIX-only module")) if importable == 'aiohttp.worker' else importable for importable in importables]

def _find_all_importables(pkg: ModuleType) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Find all importables in the project.\n\n    Return them in order.\n    '
    return sorted(set(chain.from_iterable((_discover_path_importables(Path(p), pkg.__name__) for p in pkg.__path__))))

def _discover_path_importables(pkg_pth: Path, pkg_name: str) -> Generator[str, None, None]:
    if False:
        return 10
    'Yield all importables under a given path and package.'
    for (dir_path, _d, file_names) in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)
        if pkg_dir_path.parts[-1] == '__pycache__':
            continue
        if all((Path(_).suffix != '.py' for _ in file_names)):
            continue
        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (pkg_path for (_, pkg_path, _) in pkgutil.walk_packages((str(pkg_dir_path),), prefix=f'{pkg_pref}.'))

@pytest.mark.parametrize('import_path', _mark_aiohttp_worker_for_skipping(_find_all_importables(aiohttp)))
def test_no_warnings(import_path: str) -> None:
    if False:
        return 10
    "Verify that exploding importables doesn't explode.\n\n    This is seeking for any import errors including ones caused\n    by circular imports.\n    "
    imp_cmd = (sys.executable, '-W', 'error', '-W', "ignore:module 'sre_constants' is deprecated:DeprecationWarning:pkg_resources._vendor.pyparsing", '-W', 'ignore:Creating a LegacyVersion has been deprecated and will be removed in the next major release:DeprecationWarning:', '-W', 'ignore:pkg_resources is deprecated as an API:DeprecationWarning', '-c', f'import {import_path!s}')
    subprocess.check_call(imp_cmd)
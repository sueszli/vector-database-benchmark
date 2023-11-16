from __future__ import annotations
import pytest
pytest
import os
from os.path import join, splitext
from bokeh.util.strings import nice_join
from tests.support.util.project import TOP_PATH

def test_windows_reserved_filenames() -> None:
    if False:
        i = 10
        return i + 15
    ' Certain seemingly innocuous filenames like "aux.js" will cause\n    Windows packages to fail spectacularly. This test ensures those reserved\n    names are not present in the codebase.\n\n    '
    bad: list[str] = []
    for (path, _, files) in os.walk(TOP_PATH):
        for file in files:
            if splitext(file)[0].upper() in RESERVED_NAMES:
                bad.append(join(path, file))
    assert len(bad) == 0, f'Windows reserved filenames detected:\n{nice_join(bad)}'
RESERVED_NAMES = ('CON', 'PRN', 'AUX', 'CLOCK$', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9')
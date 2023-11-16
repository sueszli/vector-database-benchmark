from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import ls_modules, verify_clean_imports
TYPING_EXTENIONS = ()
MODULES = ls_modules(skip_prefixes=TYPING_EXTENIONS)

def test_no_typing_extensions_common_combined() -> None:
    if False:
        while True:
            i = 10
    ' Basic usage of Bokeh should not result in typing_extensions being\n    imported. This test ensures that importing basic modules does not bring in\n    typing_extensions.\n\n    '
    proc = run([python, '-c', verify_clean_imports('typing_extensions', MODULES)])
    assert proc.returncode == 0, 'typing_extensions imported in common modules'
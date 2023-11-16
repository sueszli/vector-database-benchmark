from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import ls_modules, verify_clean_imports
PANDAS_ALLOWED = ('bokeh.sampledata', 'bokeh.sphinxext', 'tests.support')
MODULES = ls_modules(skip_prefixes=PANDAS_ALLOWED)

def test_no_pandas_common_combined() -> None:
    if False:
        i = 10
        return i + 15
    ' In order to keep the initial import times reasonable,  import\n    of Bokeh should not result in any Pandas code being imported. This\n    test ensures that importing basic modules does not bring in pandas.\n\n    '
    proc = run([python, '-c', verify_clean_imports('pandas', MODULES)], capture_output=True)
    output = proc.stdout.decode('utf-8').strip()
    assert proc.returncode == 0, f'pandas imported in common modules: {output}'
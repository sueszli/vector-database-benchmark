from __future__ import annotations
import pytest
pytest
from os import chdir
from subprocess import run
from tests.support.util.project import TOP_PATH

@pytest.mark.timeout(240)
def test_eslint() -> None:
    if False:
        i = 10
        return i + 15
    ' Assures that the BokehJS codebase passes configured eslint checks\n\n    '
    chdir(TOP_PATH / 'bokehjs')
    proc = run(['node', 'make', 'lint'], capture_output=True)
    assert proc.returncode == 0, f"eslint issues:\n{proc.stdout.decode('utf-8')}"
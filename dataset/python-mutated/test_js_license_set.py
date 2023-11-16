from __future__ import annotations
import pytest
pytest
import os
from subprocess import run
LICENSES = ['0BSD', 'Apache-2.0', 'AFLv2.1', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'MIT', 'Unlicense', 'WTFPL']

@pytest.mark.skip(reason="incompatible with new bokehjs' package.json setup")
def test_js_license_set() -> None:
    if False:
        return 10
    ' If the current set of JS licenses changes, they should be noted in\n    the bokehjs/LICENSE file.\n\n    '
    os.chdir('bokehjs')
    cmd = ['npx', 'license-checker', '--production', '--summary', '--onlyAllow', ';'.join(LICENSES)]
    proc = run(cmd)
    assert proc.returncode == 0, 'New BokehJS licenses detected'
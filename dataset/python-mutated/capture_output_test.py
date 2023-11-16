from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize(('s', 'version'), (pytest.param('import subprocess\nsubprocess.run(["foo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n', (3,), id='not Python3.7+'), pytest.param('from foo import run\nimport subprocess\nrun(["foo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n', (3, 7), id='run imported, but not from subprocess'), pytest.param('from foo import PIPE\nfrom subprocess import run\nsubprocess.run(["foo"], stdout=PIPE, stderr=PIPE)\n', (3, 7), id='PIPE imported, but not from subprocess'), pytest.param('from subprocess import run\nrun(["foo"], stdout=None, stderr=PIPE)\n', (3, 7), id='stdout not subprocess.PIPE')))
def test_fix_capture_output_noop(s, version):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings(min_version=version)) == s

@pytest.mark.parametrize(('s', 'expected'), (pytest.param('import subprocess\nsubprocess.run(["foo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n', 'import subprocess\nsubprocess.run(["foo"], capture_output=True)\n', id='subprocess.run and subprocess.PIPE attributes'), pytest.param('from subprocess import run, PIPE\nrun(["foo"], stdout=PIPE, stderr=PIPE)\n', 'from subprocess import run, PIPE\nrun(["foo"], capture_output=True)\n', id='run and PIPE imported from subprocess'), pytest.param('from subprocess import run, PIPE\nrun(["foo"], shell=True, stdout=PIPE, stderr=PIPE)\n', 'from subprocess import run, PIPE\nrun(["foo"], shell=True, capture_output=True)\n', id='other argument used too'), pytest.param('import subprocess\nsubprocess.run(["foo"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)\n', 'import subprocess\nsubprocess.run(["foo"], capture_output=True)\n', id='stderr used before stdout'), pytest.param('import subprocess\nsubprocess.run(stderr=subprocess.PIPE, args=["foo"], stdout=subprocess.PIPE)\n', 'import subprocess\nsubprocess.run(args=["foo"], capture_output=True)\n', id='stdout is first argument'), pytest.param('import subprocess\nsubprocess.run(\n    stderr=subprocess.PIPE, \n    args=["foo"], \n    stdout=subprocess.PIPE,\n)\n', 'import subprocess\nsubprocess.run(\n    args=["foo"], \n    capture_output=True,\n)\n', id='stdout is first argument, multiline'), pytest.param('subprocess.run(\n    "foo",\n    stdout=subprocess.PIPE,\n    stderr=subprocess.PIPE,\n    universal_newlines=True,\n)', 'subprocess.run(\n    "foo",\n    capture_output=True,\n    text=True,\n)', id='both universal_newlines and capture_output rewrite'), pytest.param('subprocess.run(\n    f"{x}(",\n    stdout=subprocess.PIPE,\n    stderr=subprocess.PIPE,\n)', 'subprocess.run(\n    f"{x}(",\n    capture_output=True,\n)', id='3.12: fstring with open brace'), pytest.param('subprocess.run(\n    f"{x})",\n    stdout=subprocess.PIPE,\n    stderr=subprocess.PIPE,\n)', 'subprocess.run(\n    f"{x})",\n    capture_output=True,\n)', id='3.12: fstring with close brace')))
def test_fix_capture_output(s, expected):
    if False:
        print('Hello World!')
    ret = _fix_plugins(s, settings=Settings(min_version=(3, 7)))
    assert ret == expected
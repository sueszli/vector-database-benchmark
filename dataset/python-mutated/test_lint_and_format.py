import os
import subprocess
from pathlib import Path
import pytest
is_CI = os.environ.get('CI') is not None

@pytest.mark.skipif(is_CI, reason='Helps contributors catch linter errors')
def test_ruff():
    if False:
        for i in range(10):
            print('nop')
    plotnine_dir = str(Path(__file__).parent.parent.absolute())
    p = subprocess.Popen(['ruff', plotnine_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, _) = p.communicate()
    s = stdout.decode('utf-8')
    msg = f'ruff found the following issues: \n\n{s}'
    assert p.returncode == 0, msg

@pytest.mark.skipif(is_CI, reason='Helps contributors catch linter errors')
def test_black():
    if False:
        i = 10
        return i + 15
    plotnine_dir = str(Path(__file__).parent.parent.absolute())
    p = subprocess.Popen(['black', plotnine_dir, '--check'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, stderr) = p.communicate()
    s = stderr.decode('utf-8')
    msg = f'black found the following issues: \n\n{s}'
    assert p.returncode == 0, msg
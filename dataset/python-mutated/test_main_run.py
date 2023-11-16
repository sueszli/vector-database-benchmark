import os
import stat
import uuid
from pathlib import Path
import pytest
from conda.common.compat import on_win
from conda.testing import CondaCLIFixture, TmpEnvFixture

def test_run_returns_int(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    with tmp_env() as prefix:
        (stdout, stderr, err) = conda_cli('run', '--prefix', prefix, 'echo', 'hi')
        assert stdout.strip() == 'hi'
        assert not stderr
        assert isinstance(err, int)

def test_run_returns_zero_errorlevel(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        print('Hello World!')
    with tmp_env() as prefix:
        (stdout, stderr, err) = conda_cli('run', '--prefix', prefix, 'exit', '0')
        assert not stdout
        assert not stderr
        assert not err

def test_run_returns_nonzero_errorlevel(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        while True:
            i = 10
    with tmp_env() as prefix:
        (stdout, stderr, err) = conda_cli('run', '--prefix', prefix, 'exit', '5')
        assert not stdout
        assert stderr
        assert err == 5

def test_run_uncaptured(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        return 10
    with tmp_env() as prefix:
        random_text = uuid.uuid4().hex
        (stdout, stderr, err) = conda_cli('run', '--prefix', prefix, '--no-capture-output', 'echo', random_text)
        assert not stdout
        assert not stderr
        assert not err

@pytest.mark.skipif(on_win, reason='cannot make readonly env on win')
def test_run_readonly_env(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        print('Hello World!')
    with tmp_env() as prefix:
        current = stat.S_IMODE(os.lstat(prefix).st_mode)
        os.chmod(prefix, current & ~stat.S_IWRITE)
        with pytest.raises(PermissionError):
            Path(prefix, 'test.txt').open('w+')
        (stdout, stderr, err) = conda_cli('run', '--prefix', prefix, 'exit', '0')
        assert not stdout
        assert not stderr
        assert not err
import pytest
from click.testing import CliRunner
from celery.app.log import Logging
from celery.bin.celery import celery

@pytest.fixture(scope='session')
def use_celery_app_trap():
    if False:
        print('Hello World!')
    return False

def test_cli(isolated_cli_runner: CliRunner):
    if False:
        print('Hello World!')
    Logging._setup = True
    res = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.app', 'beat', '-S', 't.unit.bin.proj.scheduler.mScheduler'], catch_exceptions=True)
    assert res.exit_code == 1, (res, res.stdout)
    assert res.stdout.startswith('celery beat')
    assert 'Configuration ->' in res.stdout

def test_cli_quiet(isolated_cli_runner: CliRunner):
    if False:
        i = 10
        return i + 15
    Logging._setup = True
    res = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.app', '--quiet', 'beat', '-S', 't.unit.bin.proj.scheduler.mScheduler'], catch_exceptions=True)
    assert res.exit_code == 1, (res, res.stdout)
    assert not res.stdout.startswith('celery beat')
    assert 'Configuration -> ' not in res.stdout
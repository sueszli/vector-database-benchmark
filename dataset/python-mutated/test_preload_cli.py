from click.testing import CliRunner
from celery.bin.celery import celery

def test_preload_options(isolated_cli_runner: CliRunner):
    if False:
        for i in range(10):
            print('nop')
    res_without_preload = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.app', 'purge', '-f', '--ini', 'some_ini.ini'], catch_exceptions=True)
    assert 'No such option: --ini' in res_without_preload.stdout
    assert res_without_preload.exit_code == 2
    res_without_preload = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.app', 'shell', '--ini', 'some_ini.ini'], catch_exceptions=True)
    assert 'No such option: --ini' in res_without_preload.stdout
    assert res_without_preload.exit_code == 2
    res_with_preload = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.pyramid_celery_app', 'purge', '-f', '--ini', 'some_ini.ini'], catch_exceptions=True)
    assert res_with_preload.exit_code == 0
    res_with_preload = isolated_cli_runner.invoke(celery, ['-A', 't.unit.bin.proj.pyramid_celery_app', 'shell', '--ini', 'some_ini.ini'], catch_exceptions=True)
    assert res_with_preload.exit_code == 0
import os
from click.testing import CliRunner
from lightning.app.cli.lightning_cli import run_app
from integrations_app.local import _PATH_APPS

def test_core_features_app_example():
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    result = runner.invoke(run_app, [os.path.join(_PATH_APPS, 'core_features_app', 'app.py'), '--blocking', 'False', '--open-ui', 'False', '--env', 'FOO=bar', '--env', 'BLA=bloz'], catch_exceptions=False)
    assert result.exit_code == 0
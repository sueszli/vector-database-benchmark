import json
import os
from pathlib import Path
from venv import EnvBuilder
from tests.lib import PipTestEnvironment, TestData

def test_python_interpreter(script: PipTestEnvironment, tmpdir: Path, shared_data: TestData) -> None:
    if False:
        for i in range(10):
            print('nop')
    env_path = os.fspath(tmpdir / 'venv')
    env = EnvBuilder(with_pip=False)
    env.create(env_path)
    result = script.pip('--python', env_path, 'list', '--format=json')
    before = json.loads(result.stdout)
    script.pip('--python', env_path, 'install', '-f', shared_data.find_links, '--no-index', 'simplewheel==1.0')
    result = script.pip('--python', env_path, 'list', '--format=json')
    installed = json.loads(result.stdout)
    assert {'name': 'simplewheel', 'version': '1.0'} in installed
    script.pip('--python', env_path, 'uninstall', 'simplewheel', '--yes')
    result = script.pip('--python', env_path, 'list', '--format=json')
    assert json.loads(result.stdout) == before

def test_error_python_option_wrong_location(script: PipTestEnvironment, tmpdir: Path, shared_data: TestData) -> None:
    if False:
        print('Hello World!')
    env_path = os.fspath(tmpdir / 'venv')
    env = EnvBuilder(with_pip=False)
    env.create(env_path)
    script.pip('list', '--python', env_path, '--format=json', expect_error=True)
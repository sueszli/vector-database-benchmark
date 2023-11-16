import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from venv import EnvBuilder
LOG: logging.Logger = logging.getLogger(__name__)

class AssertionError(Exception):
    pass

def production_assert(value: bool, *args: Any) -> None:
    if False:
        print('Hello World!')
    if not value:
        raise AssertionError(*args)

def validate_configuration(temporary_project_path: Path) -> None:
    if False:
        while True:
            i = 10
    configuration_path = temporary_project_path / '.pyre_configuration'
    try:
        configuration = json.loads(configuration_path.read_text())
    except json.JSONDecodeError:
        raise AssertionError(f'Invalid configuration at `{configuration_path}`')
    LOG.warning(f'Successfully created configuration at `{configuration_path}`:')
    LOG.warning(json.dumps(configuration, indent=2))
    typeshed_path = configuration.get('typeshed')
    if typeshed_path:
        typeshed_path = Path(typeshed_path)
        production_assert(typeshed_path.is_dir(), 'Explicit typeshed path is invalid.')
        production_assert((typeshed_path / 'stdlib').is_dir(), '`stdlib` was not included in typeshed.')
    binary_path = configuration.get('binary')
    if binary_path:
        binary_path = Path(binary_path)
        production_assert(binary_path.is_file(), 'Explicit binary path is invalid.')

def run_sanity_test(version: str, use_wheel: bool) -> None:
    if False:
        i = 10
        return i + 15
    message = 'wheel' if use_wheel else 'source distribution'
    LOG.warning(f'Sanity testing {message}')
    with tempfile.TemporaryDirectory() as temporary_venv:
        venv = Path(temporary_venv)
        builder = EnvBuilder(system_site_packages=False, clear=True, with_pip=True)
        builder.create(venv)
        pyre_path = venv / 'bin' / 'pyre'
        pyre_bin_path = venv / 'bin' / 'pyre.bin'
        pyre_upgrade_path = venv / 'bin' / 'pyre-upgrade'
        LOG.warning('Testing PyPi package installation...')
        wheel_flag = '--only-binary' if use_wheel else '--no-binary'
        subprocess.run([venv / 'bin' / 'pip', 'install', '--proxy=http://fwdproxy:8080/', '--index-url', 'https://test.pypi.org/simple/', '--extra-index-url', 'https://pypi.org/simple', wheel_flag, 'pyre-check', f'pyre-check=={version}'])
        production_assert(pyre_path.exists(), 'Pyre (client) was not installed.')
        production_assert(pyre_bin_path.exists(), 'Pyre binary (pyre.bin executable) was not installed.')
        production_assert(pyre_upgrade_path.exists(), 'Pyre upgrade was not installed.')
        with tempfile.TemporaryDirectory() as temporary_project:
            temporary_project_path = Path(temporary_project)
            python_file_path = temporary_project_path / 'a.py'
            python_file_path.touch()
            python_file_path.write_text('# pyre-strict \ndef foo():\n\treturn 1')
            LOG.warning('Testing `pyre init`...')
            init_process = subprocess.run([str(pyre_path), 'init'], cwd=temporary_project_path, input=b'n\n.\n', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            error_message = init_process.stderr.decode()
            production_assert(init_process.returncode == 0, f'Failed to run `pyre init` successfully: {error_message}')
            validate_configuration(temporary_project_path)
            LOG.warning('Testing `pyre` error reporting...')
            result = subprocess.run([pyre_path, '--binary', pyre_bin_path, '--output=json', 'check'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=temporary_project_path)
            try:
                errors = json.loads(result.stdout)
            except json.JSONDecodeError:
                error_message = result.stderr.decode()
                raise AssertionError(f'Pyre did not successfully finish type checking: {error_message}')
            production_assert(errors and errors[0]['name'] == 'Missing return annotation', 'Incorrect pyre errors returned.' if errors else 'Expected pyre errors but none returned.')
            LOG.warning('Testing `pyre upgrade`...')
            upgrade_process = subprocess.run([str(pyre_upgrade_path), 'fixme'], cwd=temporary_project_path, input=b'[]', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            error_message = upgrade_process.stderr.decode()
            production_assert(upgrade_process.returncode == 0, f'Failed to run `pyre-upgrade` successfully: {error_message}')

def main() -> None:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Test wheel & source distribution for basic functionality.')
    parser.add_argument('version', type=str)
    arguments = parser.parse_args()
    version: str = arguments.version
    run_sanity_test(version, use_wheel=True)
if __name__ == '__main__':
    main()
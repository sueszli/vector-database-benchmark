"""Install Python development dependencies."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from typing import List, Optional
INSTALLATION_TOOL_VERSIONS = {'pip': '23.1.2', 'pip-tools': '6.13.0', 'setuptools': '67.7.1'}
REQUIREMENTS_DEV_FILE_PATH = 'requirements_dev.in'
COMPILED_REQUIREMENTS_DEV_FILE_PATH = 'requirements_dev.txt'
_PARSER = argparse.ArgumentParser('Install Python development dependencies')
_PARSER.add_argument('--assert_compiled', action='store_true', help='Assert that the dev requirements file is already compiled.')
_PARSER.add_argument('--uninstall', action='store_true', help='Uninstall all dev requirements.')

def check_python_env_is_suitable() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Raise an error if we are not in a virtual environment or on CI.\n\n    We want developers to use a virtual environment when developing locally so\n    that our scripts don\'t change their global Python environments. On CI\n    however, it\'s okay to change the global environment since the checks are\n    running in an ephemeral virtual machine. Therefore, a "suitable" Python\n    environment is one that either is on CI or is a virtual environment.\n    '
    if 'GITHUB_ACTION' in os.environ:
        return
    if sys.prefix == sys.base_prefix and (not (hasattr(sys, 'real_prefix') and getattr(sys, 'real_prefix'))):
        raise AssertionError('Oppia must be developed within a virtual environment.')

def install_installation_tools() -> None:
    if False:
        while True:
            i = 10
    'Install the minimal tooling needed to install dependencies.'
    for (package, version) in INSTALLATION_TOOL_VERSIONS.items():
        subprocess.run([sys.executable, '-m', 'pip', 'install', f'{package}=={version}'], check=True, encoding='utf-8')

def install_dev_dependencies() -> None:
    if False:
        return 10
    'Install dev dependencies from COMPILED_REQUIREMENTS_DEV_FILE_PATH.'
    subprocess.run(['pip-sync', COMPILED_REQUIREMENTS_DEV_FILE_PATH, '--pip-args', '--require-hashes --no-deps'], check=True, encoding='utf-8')

def uninstall_dev_dependencies() -> None:
    if False:
        return 10
    'Uninstall dev dependencies from COMPILED_REQUIREMENTS_DEV_FILE_PATH.'
    subprocess.run(['pip', 'uninstall', '-r', COMPILED_REQUIREMENTS_DEV_FILE_PATH, '-y'], check=True, encoding='utf-8')

def compile_pip_requirements(requirements_path: str, compiled_path: str) -> bool:
    if False:
        while True:
            i = 10
    'Compile a requirements.txt file.\n\n    Args:\n        requirements_path: str. Path to the requirements.in file.\n        compiled_path: str. Path to the requirements.txt file.\n\n    Returns:\n        bool. Whether the compiled dev requirements file was changed.\n    '
    with open(compiled_path, 'r', encoding='utf-8') as f:
        old_compiled = f.read()
    subprocess.run(['pip-compile', '--no-emit-index-url', '--generate-hashes', requirements_path, '--output-file', compiled_path], check=True, encoding='utf-8')
    with open(compiled_path, 'r', encoding='utf-8') as f:
        new_compiled = f.read()
    return old_compiled != new_compiled

def main(cli_args: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Install all dev dependencies.'
    args = _PARSER.parse_args(cli_args)
    check_python_env_is_suitable()
    install_installation_tools()
    not_compiled = compile_pip_requirements(REQUIREMENTS_DEV_FILE_PATH, COMPILED_REQUIREMENTS_DEV_FILE_PATH)
    if args.uninstall:
        uninstall_dev_dependencies()
    else:
        install_dev_dependencies()
        if args.assert_compiled and not_compiled:
            raise RuntimeError(f'The Python development requirements file {COMPILED_REQUIREMENTS_DEV_FILE_PATH} was changed by the installation script. Please commit the changes. You can get the changes again by running this command: python -m scripts.install_python_dev_dependencies')
if __name__ == '__main__':
    main()
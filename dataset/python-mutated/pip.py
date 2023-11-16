from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.exceptions import PoetryException
from poetry.utils.env import EnvCommandError
if TYPE_CHECKING:
    from pathlib import Path
    from poetry.utils.env import Env

def pip_install(path: Path, environment: Env, editable: bool=False, deps: bool=False, upgrade: bool=False) -> str:
    if False:
        for i in range(10):
            print('nop')
    is_wheel = path.suffix == '.whl'
    args = ['install', '--disable-pip-version-check', '--isolated', '--no-input', '--prefix', str(environment.path)]
    if not is_wheel and (not editable):
        args.insert(1, '--use-pep517')
    if upgrade:
        args.append('--upgrade')
    if not deps:
        args.append('--no-deps')
    if editable:
        if not path.is_dir():
            raise PoetryException('Cannot install non directory dependencies in editable mode')
        args.append('-e')
    args.append(str(path))
    try:
        return environment.run_pip(*args)
    except EnvCommandError as e:
        raise PoetryException(f'Failed to install {path}') from e
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from tox.config.loader.memory import MemoryLoader
from tox.plugin import impl
from tox.report import HandledError
from tox.session.cmd.run.common import env_run_create_flags
from tox.session.cmd.run.sequential import run_sequential
from tox.session.env_select import CliEnv, register_env_select_flags
if TYPE_CHECKING:
    from tox.config.cli.parser import ToxParser
    from tox.session.state import State

@impl
def tox_add_option(parser: ToxParser) -> None:
    if False:
        for i in range(10):
            print('nop')
    help_msg = 'sets up a development environment at ENVDIR based on the tox configuration specified '
    our = parser.add_command('devenv', ['d'], help_msg, devenv)
    our.add_argument('devenv_path', metavar='path', default=Path('venv'), nargs='?', type=Path)
    register_env_select_flags(our, default=CliEnv('py'), multiple=False)
    env_run_create_flags(our, mode='devenv')

def devenv(state: State) -> int:
    if False:
        i = 10
        return i + 15
    opt = state.conf.options
    opt.devenv_path = opt.devenv_path.absolute()
    opt.skip_missing_interpreters = False
    opt.no_test = False
    opt.package_only = False
    opt.install_pkg = None
    opt.skip_pkg_install = False
    opt.no_test = True
    loader = MemoryLoader(usedevelop=True, env_dir=opt.devenv_path)
    state.conf.memory_seed_loaders[next(iter(opt.env))].append(loader)
    state.envs.ensure_only_run_env_is_active()
    envs = list(state.envs.iter())
    if len(envs) != 1:
        msg = f"exactly one target environment allowed in devenv mode but found {', '.join(envs)}"
        raise HandledError(msg)
    result = run_sequential(state)
    if result == 0:
        logging.warning('created development environment under %s', opt.devenv_path)
    return result
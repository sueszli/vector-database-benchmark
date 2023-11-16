from __future__ import annotations
import sys
from os import environ
from typing import TYPE_CHECKING
from typing import cast
from poetry.console.commands.env_command import EnvCommand
if TYPE_CHECKING:
    from poetry.utils.env import VirtualEnv

class ShellCommand(EnvCommand):
    name = 'shell'
    description = 'Spawns a shell within the virtual environment.'
    help = "The <info>shell</> command spawns a shell, according to the\n<comment>$SHELL</> environment variable, within the virtual environment.\nIf one doesn't exist yet, it will be created.\n"

    def handle(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        from poetry.utils.shell import Shell
        if self._is_venv_activated():
            self.line(f'Virtual environment already activated: <info>{self.env.path}</>')
            return 0
        self.line(f'Spawning shell within <info>{self.env.path}</>')
        env = self.env
        assert env.is_venv()
        env = cast('VirtualEnv', env)
        environ['POETRY_ACTIVE'] = '1'
        shell = Shell.get()
        shell.activate(env)
        environ.pop('POETRY_ACTIVE')
        return 0

    def _is_venv_activated(self) -> bool:
        if False:
            print('Hello World!')
        return bool(environ.get('POETRY_ACTIVE')) or getattr(sys, 'real_prefix', sys.prefix) == str(self.env.path)
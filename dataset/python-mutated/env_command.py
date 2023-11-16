from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.console.commands.command import Command
if TYPE_CHECKING:
    from poetry.utils.env import Env

class EnvCommand(Command):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._env: Env | None = None
        super().__init__()

    @property
    def env(self) -> Env:
        if False:
            while True:
                i = 10
        assert self._env is not None
        return self._env

    def set_env(self, env: Env) -> None:
        if False:
            while True:
                i = 10
        self._env = env
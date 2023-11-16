from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.console.commands.command import Command
if TYPE_CHECKING:
    from collections.abc import Callable

class AboutCommand(Command):
    name = 'about'
    description = 'Shows information about Poetry.'

    def handle(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        from poetry.utils._compat import metadata
        version: Callable[[str], str] = metadata.version
        self.line(f"<info>Poetry - Package Management for Python\n\nVersion: {version('poetry')}\nPoetry-Core Version: {version('poetry-core')}</info>\n\n<comment>Poetry is a dependency manager tracking local dependencies of your projects and libraries.\nSee <fg=blue>https://github.com/python-poetry/poetry</> for more information.</comment>")
        return 0
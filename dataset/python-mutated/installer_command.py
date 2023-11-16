from __future__ import annotations
from typing import TYPE_CHECKING
from poetry.console.commands.env_command import EnvCommand
from poetry.console.commands.group_command import GroupCommand
if TYPE_CHECKING:
    from poetry.installation.installer import Installer

class InstallerCommand(GroupCommand, EnvCommand):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._installer: Installer | None = None
        super().__init__()

    def reset_poetry(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().reset_poetry()
        self.installer.set_package(self.poetry.package)
        self.installer.set_locker(self.poetry.locker)

    @property
    def installer(self) -> Installer:
        if False:
            return 10
        assert self._installer is not None
        return self._installer

    def set_installer(self, installer: Installer) -> None:
        if False:
            print('Hello World!')
        self._installer = installer
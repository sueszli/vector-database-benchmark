from __future__ import annotations
from poetry.core.packages.dependency_group import MAIN_GROUP
from poetry.console.commands.install import InstallCommand
from poetry.console.commands.self.self_command import SelfCommand

class SelfInstallCommand(SelfCommand, InstallCommand):
    name = 'self install'
    description = 'Install locked packages (incl. addons) required by this Poetry installation.'
    options = [o for o in InstallCommand.options if o.name in {'sync', 'dry-run'}]
    help = f'The <c1>self install</c1> command ensures all additional packages specified are installed in the current runtime environment.\n\nThis is managed in the <comment>{SelfCommand.get_default_system_pyproject_file()}</> file.\n\nYou can add more packages using the <c1>self add</c1> command and remove them using the <c1>self remove</c1> command.\n'

    @property
    def activated_groups(self) -> set[str]:
        if False:
            i = 10
            return i + 15
        return {MAIN_GROUP, self.default_group}
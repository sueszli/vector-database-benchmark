import abc
import re
from pdm.project import Config, Project
from pdm.termui import Verbosity

class BaseFixer(abc.ABC):
    """Base class for fixers"""
    identifier: str
    breaking: bool = False

    def __init__(self, project: Project) -> None:
        if False:
            while True:
                i = 10
        self.project = project

    def log(self, message: str, verbosity: Verbosity=Verbosity.DETAIL) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.project.core.ui.echo(message, verbosity=verbosity)

    @abc.abstractmethod
    def get_message(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return a description of the problem'

    @abc.abstractmethod
    def fix(self) -> None:
        if False:
            i = 10
            return i + 15
        'Perform the fix'

    @abc.abstractmethod
    def check(self) -> bool:
        if False:
            return 10
        'Check if the problem exists'

class ProjectConfigFixer(BaseFixer):
    """Fix the project config"""
    identifier = 'project-config'

    def get_message(self) -> str:
        if False:
            i = 10
            return i + 15
        return '[success]python.path[/] config needs to be moved to [info].pdm-python[/] and [info].pdm.toml[/] needs to be renamed to [info]pdm.toml[/]'

    def _fix_gitignore(self) -> None:
        if False:
            while True:
                i = 10
        gitignore = self.project.root.joinpath('.gitignore')
        if not gitignore.exists():
            return
        content = gitignore.read_text('utf8')
        if '.pdm-python' not in content:
            content = re.sub('^\\.pdm\\.toml$', '.pdm-python', content, flags=re.M)
            gitignore.write_text(content, 'utf8')

    def fix(self) -> None:
        if False:
            i = 10
            return i + 15
        old_file = self.project.root.joinpath('.pdm.toml')
        config = Config(old_file).self_data
        if not self.project.root.joinpath('.pdm-python').exists() and config.get('python.path'):
            self.log('Creating .pdm-python...', verbosity=Verbosity.DETAIL)
            self.project.root.joinpath('.pdm-python').write_text(config['python.path'])
        self.project.project_config
        self.log('Moving .pdm.toml to pdm.toml...', verbosity=Verbosity.DETAIL)
        old_file.unlink()
        self.log('Fixing .gitignore...', verbosity=Verbosity.DETAIL)
        self._fix_gitignore()

    def check(self) -> bool:
        if False:
            print('Hello World!')
        return self.project.root.joinpath('.pdm.toml').exists()
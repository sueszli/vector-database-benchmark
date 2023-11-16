import logging
from distutils import core as distutils_core
from importlib import reload
from pathlib import Path
from typing import Dict, Optional, Set
import pathspec
from pkg_resources import Requirement, parse_requirements
from dagster_buildkite.git import ChangedFiles, GitInfo
changed_filetypes = ['.py', '.cfg', '.toml', '.yaml', '.ipynb', '.yml', '.ini', '.jinja']

class PythonPackage:

    def __init__(self, setup_py_path: Path):
        if False:
            return 10
        self.directory = setup_py_path.parent
        reload(distutils_core)
        distribution = distutils_core.run_setup(str(setup_py_path), stop_after='init')
        self._install_requires = distribution.install_requires
        self._extras_require = distribution.extras_require
        self.name = distribution.get_name()

    @property
    def install_requires(self) -> Set[Requirement]:
        if False:
            i = 10
            return i + 15
        return set((requirement for requirement in parse_requirements(self._install_requires) if PythonPackages.get(requirement.name)))

    @property
    def extras_require(self) -> Dict[str, Set[Requirement]]:
        if False:
            while True:
                i = 10
        extras_require = {}
        for (extra, requirements) in self._extras_require.items():
            extras_require[extra] = set((requirement for requirement in parse_requirements(requirements) if PythonPackages.get(requirement.name)))
        return extras_require

    def __str__(self):
        if False:
            print('Hello World!')
        return self.name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'PythonPackage({self.name})'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.directory == other.directory

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.directory)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self.name < other.name

class PythonPackages:
    _repositories: Set[Path] = set()
    all: Dict[str, PythonPackage] = dict()
    with_changes: Set[PythonPackage] = set()

    @classmethod
    def get(cls, name: str) -> Optional[PythonPackage]:
        if False:
            i = 10
            return i + 15
        return cls.all.get(name) or cls.all.get(name.replace('_', '-')) or cls.all.get(name.replace('-', '_'))

    @classmethod
    def walk_dependencies(cls, requirement: Requirement) -> Set[PythonPackage]:
        if False:
            print('Hello World!')
        dependencies: Set[PythonPackage] = set()
        dagster_package = cls.get(requirement.name)
        if not dagster_package:
            return dependencies
        dependencies.add(dagster_package)
        for extra in requirement.extras:
            for req in dagster_package.extras_require.get(extra, set()):
                dependencies.update(cls.walk_dependencies(req))
        for req in dagster_package.install_requires:
            dependencies.update(cls.walk_dependencies(req))
        return dependencies

    @classmethod
    def load_from_git(cls, git_info: GitInfo) -> None:
        if False:
            i = 10
            return i + 15
        if git_info.directory in cls._repositories:
            return None
        ChangedFiles.load_from_git(git_info)
        logging.info('Finding Python packages:')
        git_ignore = git_info.directory / '.gitignore'
        if git_ignore.exists():
            ignored = git_ignore.read_text().splitlines()
            git_ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', ignored)
        else:
            git_ignore_spec = pathspec.PathSpec([])
        packages = set([PythonPackage(Path(setup)) for setup in git_info.directory.rglob('setup.py') if not git_ignore_spec.match_file(str(setup))])
        for package in sorted(packages):
            logging.info('  - ' + package.name)
            cls.all[package.name] = package
        packages_with_changes: Set[PythonPackage] = set()
        logging.info('Finding changed packages:')
        for package in packages:
            for change in ChangedFiles.all:
                if change in package.directory.rglob('*') and change.suffix in changed_filetypes and ('_tests/' not in str(change)):
                    packages_with_changes.add(package)
        for package in sorted(packages_with_changes):
            logging.info('  - ' + package.name)
            cls.with_changes.add(package)
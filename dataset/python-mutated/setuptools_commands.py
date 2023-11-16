import glob
import os
import sys
from typing import Any, Dict, Iterator, List
from warnings import warn
import setuptools
from . import api
from .settings import DEFAULT_CONFIG

class ISortCommand(setuptools.Command):
    """The :class:`ISortCommand` class is used by setuptools to perform
    imports checks on registered modules.
    """
    description = 'Run isort on modules registered in setuptools'
    user_options: List[Any] = []

    def initialize_options(self) -> None:
        if False:
            while True:
                i = 10
        default_settings = vars(DEFAULT_CONFIG).copy()
        for (key, value) in default_settings.items():
            setattr(self, key, value)

    def finalize_options(self) -> None:
        if False:
            i = 10
            return i + 15
        'Get options from config files.'
        self.arguments: Dict[str, Any] = {}
        self.arguments['settings_path'] = os.getcwd()

    def distribution_files(self) -> Iterator[str]:
        if False:
            while True:
                i = 10
        'Find distribution packages.'
        if self.distribution.packages:
            package_dirs = self.distribution.package_dir or {}
            for package in self.distribution.packages:
                pkg_dir = package
                if package in package_dirs:
                    pkg_dir = package_dirs[package]
                elif '' in package_dirs:
                    pkg_dir = package_dirs[''] + os.path.sep + pkg_dir
                yield pkg_dir.replace('.', os.path.sep)
        if self.distribution.py_modules:
            for filename in self.distribution.py_modules:
                yield f'{filename}.py'
        yield 'setup.py'

    def run(self) -> None:
        if False:
            return 10
        arguments = self.arguments
        wrong_sorted_files = False
        for path in self.distribution_files():
            for python_file in glob.iglob(os.path.join(path, '*.py')):
                try:
                    if not api.check_file(python_file, **arguments):
                        wrong_sorted_files = True
                except OSError as error:
                    warn(f'Unable to parse file {python_file} due to {error}')
        if wrong_sorted_files:
            sys.exit(1)
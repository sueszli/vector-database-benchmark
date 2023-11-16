import ast
import subprocess
import unittest
from pathlib import Path
from textwrap import dedent
from typing import List
from unittest.mock import call, MagicMock, patch
from ..filesystem import add_local_mode, Filesystem, LocalMode, MercurialBackedFilesystem, Target, TargetCollector

class FilesystemTest(unittest.TestCase):

    def assert_collector(self, source: str, expected_targets: List[Target], pyre_only: bool) -> None:
        if False:
            while True:
                i = 10
        target_collector = TargetCollector(pyre_only)
        tree = ast.parse(dedent(source))
        target_collector.visit(tree)
        targets = target_collector.result()
        self.assertEqual(expected_targets, targets)

    def test_target_collector(self) -> None:
        if False:
            return 10
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = []
        self.assert_collector(source, expected_targets, False)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            typing = "False",\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=False, pyre=True, check_types=False)]
        self.assert_collector(source, expected_targets, False)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            check_types_options = "strict",\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=True, pyre=True, check_types=False)]
        self.assert_collector(source, expected_targets, False)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            check_types = True,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n\n        python_unittest(\n            name = "test_target_name",\n            srcs = glob([\n                "**/tests/*.py",\n            ]),\n            check_types = False,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=False, pyre=True, check_types=True), Target('test_target_name', strict=False, pyre=True, check_types=False)]
        self.assert_collector(source, expected_targets, False)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            check_types = True,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n\n        python_unittest(\n            name = "test_target_name",\n            srcs = glob([\n                "**/tests/*.py",\n            ]),\n            check_types = True,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=False, pyre=True, check_types=True), Target('test_target_name', strict=False, pyre=True, check_types=True)]
        self.assert_collector(source, expected_targets, False)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            check_types = True,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n\n        python_unittest(\n            name = "test_target_name",\n            srcs = glob([\n                "**/tests/*.py",\n            ]),\n            check_types = True,\n            check_types_options = "mypy",\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=False, pyre=True, check_types=True)]
        self.assert_collector(source, expected_targets, True)
        source = '\n        load("@path:python_binary.bzl", "python_binary")\n\n        python_binary(\n            name = "target_name",\n            main_module = "path.to.module",\n            check_types = True,\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n\n        python_unittest(\n            name = "test_target_name",\n            srcs = glob([\n                "**/tests/*.py",\n            ]),\n            check_types = True,\n            check_types_options = "strict, mypy",\n            deps = [\n                ":dependency_target_name",\n            ],\n        )\n        '
        expected_targets = [Target('target_name', strict=False, pyre=True, check_types=True)]
        self.assert_collector(source, expected_targets, True)

    def test_filesystem_list_bare(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        filesystem = Filesystem()
        with patch.object(subprocess, 'run') as run:
            filesystem.list('.', ['.pyre_configuration.local'])
            run.assert_has_calls([call(['find', '.', '(', '-path', './.pyre_configuration.local', ')'], stdout=subprocess.PIPE, cwd='.'), call().stdout.decode('utf-8'), call().stdout.decode().split()])
        with patch.object(subprocess, 'run') as run:
            filesystem.list('/root', ['**/*.py', 'foo.cpp'], exclude=['bar/*.py'])
            run.assert_has_calls([call(['find', '.', '(', '-path', './**/*.py', '-or', '-path', './foo.cpp', ')', '-and', '!', '(', '-path', './bar/*.py', ')'], stdout=subprocess.PIPE, cwd='/root'), call().stdout.decode('utf-8'), call().stdout.decode().split()])

        def fail_command(*args: object, **kwargs: object) -> 'subprocess.CompletedProcess[bytes]':
            if False:
                return 10
            return subprocess.CompletedProcess(args=[], returncode=1, stdout=''.encode('utf-8'))
        with patch.object(subprocess, 'run') as run:
            run.side_effect = fail_command
            self.assertEqual([], filesystem.list('.', ['.pyre_configuration.local']))
            run.assert_has_calls([call(['find', '.', '(', '-path', './.pyre_configuration.local', ')'], stdout=subprocess.PIPE, cwd='.')])

    def test_filesystem_list_mercurial(self) -> None:
        if False:
            return 10
        filesystem = MercurialBackedFilesystem()
        with patch.object(subprocess, 'run') as run:
            filesystem.list('.', ['.pyre_configuration.local'])
            run.assert_has_calls([call(['hg', 'files', '--include', '.pyre_configuration.local'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd='.'), call().stdout.decode('utf-8'), call().stdout.decode().split()])
        with patch.object(subprocess, 'run') as run:
            filesystem.list('/root', ['**/*.py', 'foo.cpp'], exclude=['bar/*.py'])
            run.assert_has_calls([call(['hg', 'files', '--include', '**/*.py', '--include', 'foo.cpp', '--exclude', 'bar/*.py'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd='/root'), call().stdout.decode('utf-8'), call().stdout.decode().split()])

        def fail_command(*args: object, **kwargs: object) -> 'subprocess.CompletedProcess[bytes]':
            if False:
                return 10
            return subprocess.CompletedProcess(args=[], returncode=1, stdout=''.encode('utf-8'))
        with patch.object(subprocess, 'run') as run:
            run.side_effect = fail_command
            self.assertEqual([], filesystem.list('.', ['.pyre_configuration.local']))
            run.assert_has_calls([call(['hg', 'files', '--include', '.pyre_configuration.local'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd='.')])

    @patch.object(Path, 'read_text')
    def test_add_local_mode(self, read_text: MagicMock) -> None:
        if False:
            for i in range(10):
                print('nop')
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '1\n2'
            add_local_mode('local.py', LocalMode.UNSAFE)
            path_write_text.assert_called_once_with('# pyre-unsafe\n1\n2')
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '# comment\n# comment\n1'
            add_local_mode('local.py', LocalMode.UNSAFE)
            path_write_text.assert_called_once_with('# comment\n# comment\n\n# pyre-unsafe\n1')
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '# comment\n# pyre-strict\n1'
            add_local_mode('local.py', LocalMode.UNSAFE)
            path_write_text.assert_not_called()
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '# comment\n# pyre-ignore-all-errors\n1'
            add_local_mode('local.py', LocalMode.UNSAFE)
            path_write_text.assert_not_called()
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '1\n2'
            add_local_mode('local.py', LocalMode.STRICT)
            path_write_text.assert_called_once_with('# pyre-strict\n1\n2')
        with patch.object(Path, 'write_text') as path_write_text:
            read_text.return_value = '1\n2'
            add_local_mode('local.py', LocalMode.IGNORE)
            path_write_text.assert_called_once_with('# pyre-ignore-all-errors\n1\n2')
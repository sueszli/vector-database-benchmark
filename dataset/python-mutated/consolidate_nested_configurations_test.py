import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, MagicMock, patch
from ... import errors
from ...repository import Repository
from .. import consolidate_nested_configurations
from ..consolidate_nested_configurations import consolidate_nested, ConsolidateNestedConfigurations
repository = Repository()

class ConsolidateNestedConfigurationsTest(unittest.TestCase):

    def test_gather_nested_configuration_mapping(self) -> None:
        if False:
            print('Hello World!')
        arguments = MagicMock()
        configurations = []
        expected_mapping = {}
        mapping = ConsolidateNestedConfigurations.from_arguments(arguments, repository).gather_nested_configuration_mapping(configurations)
        self.assertEqual(expected_mapping, mapping)
        configurations = ['a/.pyre_configuration.local', 'b/.pyre_configuration.local', 'a/b/.pyre_configuration.local', 'aa/.pyre_configuration.local']
        expected_mapping = {'a/.pyre_configuration.local': ['a/b/.pyre_configuration.local'], 'aa/.pyre_configuration.local': [], 'b/.pyre_configuration.local': []}
        mapping = ConsolidateNestedConfigurations.from_arguments(arguments, repository).gather_nested_configuration_mapping(configurations)
        self.assertEqual(expected_mapping, mapping)

    @patch(f'{consolidate_nested_configurations.__name__}.Repository.remove_paths')
    @patch(f'{consolidate_nested_configurations.__name__}.Configuration.get_errors')
    def test_consolidate(self, get_errors, remove_paths) -> None:
        if False:
            return 10
        get_errors.return_value = errors.Errors([])
        with tempfile.TemporaryDirectory() as root:
            configuration_path = os.path.join(root, '.pyre_configuration.local')
            nested_a = tempfile.mkdtemp('a', dir=root)
            nested_b = tempfile.mkdtemp('b', dir=root)
            nested_a_configuration = os.path.join(nested_a, '.pyre_configuration.local')
            nested_b_configuration = os.path.join(nested_b, '.pyre_configuration.local')
            with open(configuration_path, 'w+') as configuration_file, open(nested_a_configuration, 'w+') as nested_configuration_a, open(nested_b_configuration, 'w+') as nested_configuration_b:
                json.dump({'targets': ['//x/...']}, configuration_file)
                json.dump({'targets': ['//a/...']}, nested_configuration_a)
                json.dump({'targets': ['//b/...']}, nested_configuration_b)
                configuration_file.seek(0)
                nested_configuration_a.seek(0)
                nested_configuration_b.seek(0)
                consolidate_nested(repository, Path(configuration_path), [Path(nested_a_configuration), Path(nested_b_configuration)])
                remove_paths.assert_has_calls([call([Path(nested_a_configuration)]), call([Path(nested_b_configuration)])])
                self.assertEqual(json.load(configuration_file), {'targets': ['//x/...', '//a/...', '//b/...']})

    @patch(f'{consolidate_nested_configurations.__name__}.Repository.commit_changes')
    @patch(f'{consolidate_nested_configurations.__name__}.consolidate_nested')
    def test_run_skip(self, consolidate, commit_changes) -> None:
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as root:
            arguments = MagicMock()
            arguments.subdirectory = root
            arguments.lint = False
            arguments.no_commit = False
            ConsolidateNestedConfigurations.from_arguments(arguments, repository).run()
            consolidate.assert_not_called()
            with open(os.path.join(root, '.pyre_configuration.local'), 'w+'):
                ConsolidateNestedConfigurations.from_arguments(arguments, repository).run()
                consolidate.assert_not_called()

    @patch(f'{consolidate_nested_configurations.__name__}.Repository.commit_changes')
    @patch(f'{consolidate_nested_configurations.__name__}.consolidate_nested')
    @patch(f'{consolidate_nested_configurations.__name__}.Configuration.get_errors')
    def test_run_topmost(self, get_errors, consolidate, commit_changes) -> None:
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as root:
            arguments = MagicMock()
            arguments.subdirectory = root
            arguments.lint = False
            arguments.no_commit = False
            subdirectory_a = tempfile.mkdtemp('a', dir=root)
            subdirectory_b = tempfile.mkdtemp('b', dir=root)
            with open(os.path.join(root, '.pyre_configuration.local'), 'w+') as configuration, open(os.path.join(subdirectory_a, '.pyre_configuration.local'), 'w+') as nested_a, open(os.path.join(subdirectory_b, '.pyre_configuration.local'), 'w+') as nested_b:
                json.dump({'targets': ['//x/...']}, configuration)
                configuration.seek(0)
                ConsolidateNestedConfigurations.from_arguments(arguments, repository).run()
                consolidate.assert_called_once_with(repository, Path(configuration.name), sorted([Path(nested_a.name), Path(nested_b.name)]))

    @patch(f'{consolidate_nested_configurations.__name__}.Repository.commit_changes')
    @patch(f'{consolidate_nested_configurations.__name__}.consolidate_nested')
    @patch(f'{consolidate_nested_configurations.__name__}.Configuration.get_errors')
    def test_run_no_topmost(self, get_errors, consolidate, commit_changes) -> None:
        if False:
            return 10
        with tempfile.TemporaryDirectory() as root:
            arguments = MagicMock()
            arguments.subdirectory = root
            arguments.lint = False
            arguments.no_commit = False
            subdirectory_a = tempfile.mkdtemp('a', dir=root)
            subdirectory_b = tempfile.mkdtemp('b', dir=root)
            with open(os.path.join(subdirectory_a, '.pyre_configuration.local'), 'w+'), open(os.path.join(subdirectory_b, '.pyre_configuration.local'), 'w+'):
                ConsolidateNestedConfigurations.from_arguments(arguments, repository).run()
                consolidate.assert_not_called()
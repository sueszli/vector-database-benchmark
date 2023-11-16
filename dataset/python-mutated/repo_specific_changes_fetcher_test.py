"""Unit tests for scripts/release_scripts/repo_specific_changes_fetcher.py."""
from __future__ import annotations
import builtins
import os
from core.tests import test_utils
from scripts import common
from scripts.release_scripts import repo_specific_changes_fetcher
from typing import Dict, Final, List
RELEASE_TEST_DIR: Final = os.path.join('core', 'tests', 'release_sources', '')
MOCK_FECONF_FILEPATH: Final = os.path.join(RELEASE_TEST_DIR, 'feconf.txt')

class GetRepoSpecificChangesTest(test_utils.GenericTestBase):
    """Test the methods for obtaining repo specific changes."""

    def test_get_changed_schema_version_constant_names_with_no_diff(self) -> None:
        if False:
            i = 10
            return i + 15

        def mock_run_cmd(unused_cmd: str) -> str:
            if False:
                i = 10
                return i + 15
            return 'CURRENT_STATE_SCHEMA_VERSION = 3\nCURRENT_COLLECTION_SCHEMA_VERSION = 4\n'
        run_cmd_swap = self.swap(common, 'run_cmd', mock_run_cmd)
        feconf_swap = self.swap(repo_specific_changes_fetcher, 'FECONF_FILEPATH', MOCK_FECONF_FILEPATH)
        with run_cmd_swap, feconf_swap:
            actual_version_changes = repo_specific_changes_fetcher.get_changed_schema_version_constant_names('release_tag')
        self.assertEqual(actual_version_changes, [])

    def test_get_changed_schema_version_constant_names_with_diff(self) -> None:
        if False:
            return 10

        def mock_run_cmd(unused_cmd: str) -> str:
            if False:
                i = 10
                return i + 15
            return 'CURRENT_STATE_SCHEMA_VERSION = 8\nCURRENT_COLLECTION_SCHEMA_VERSION = 4\n'
        run_cmd_swap = self.swap(common, 'run_cmd', mock_run_cmd)
        feconf_swap = self.swap(repo_specific_changes_fetcher, 'FECONF_FILEPATH', MOCK_FECONF_FILEPATH)
        with run_cmd_swap, feconf_swap:
            actual_version_changes = repo_specific_changes_fetcher.get_changed_schema_version_constant_names('release_tag')
        self.assertEqual(actual_version_changes, ['CURRENT_STATE_SCHEMA_VERSION'])

    def test_get_setup_scripts_changes_status_to_get_changed_scripts_status(self) -> None:
        if False:
            while True:
                i = 10

        def mock_run_cmd(unused_cmd: str) -> str:
            if False:
                i = 10
                return i + 15
            return 'scripts/setup.py\nscripts/setup_gae.py'
        with self.swap(common, 'run_cmd', mock_run_cmd):
            actual_scripts = repo_specific_changes_fetcher.get_setup_scripts_changes_status('release_tag')
        expected_scripts = {'scripts/setup.py': True, 'scripts/setup_gae.py': True, 'scripts/install_third_party_libs.py': False, 'scripts/install_third_party.py': False}
        self.assertEqual(actual_scripts, expected_scripts)

    def test_get_changed_storage_models_filenames(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def mock_run_cmd(unused_cmd: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'scripts/setup.py\nextensions/test.ts\ncore/storage/activity/gae_models.py\ncore/storage/user/gae_models.py'
        with self.swap(common, 'run_cmd', mock_run_cmd):
            actual_storgae_models = repo_specific_changes_fetcher.get_changed_storage_models_filenames('release_tag')
        expected_storage_models = ['core/storage/activity/gae_models.py', 'core/storage/user/gae_models.py']
        self.assertEqual(actual_storgae_models, expected_storage_models)

    def test_unmodified_state_shows_no_change_in_code_files(self) -> None:
        if False:
            while True:
                i = 10

        def mock_get_changed_schema_version_constant_names(unused_release_tag_to_diff_against: str) -> None:
            if False:
                i = 10
                return i + 15
            return None

        def mock_get_setup_scripts_changes_status(unused_release_tag_to_diff_against: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            return None

        def mock_get_changed_storage_models_filenames(unused_release_tag_to_diff_against: str) -> None:
            if False:
                return 10
            return None
        versions_swap = self.swap(repo_specific_changes_fetcher, 'get_changed_schema_version_constant_names', mock_get_changed_schema_version_constant_names)
        setup_scripts_swap = self.swap(repo_specific_changes_fetcher, 'get_setup_scripts_changes_status', mock_get_setup_scripts_changes_status)
        storage_models_swap = self.swap(repo_specific_changes_fetcher, 'get_changed_storage_models_filenames', mock_get_changed_storage_models_filenames)
        with versions_swap, setup_scripts_swap, storage_models_swap:
            expected_changes: List[str] = []
            self.assertEqual(repo_specific_changes_fetcher.get_changes('release_tag'), expected_changes)

    def test_get_changes(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def mock_get_changed_schema_version_constant_names(unused_release_tag_to_diff_against: str) -> List[str]:
            if False:
                while True:
                    i = 10
            return ['version_change']

        def mock_get_setup_scripts_changes_status(unused_release_tag_to_diff_against: str) -> Dict[str, bool]:
            if False:
                i = 10
                return i + 15
            return {'setup_changes': True}

        def mock_get_changed_storage_models_filenames(unused_release_tag_to_diff_against: str) -> List[str]:
            if False:
                print('Hello World!')
            return ['storage_changes']
        versions_swap = self.swap(repo_specific_changes_fetcher, 'get_changed_schema_version_constant_names', mock_get_changed_schema_version_constant_names)
        setup_scripts_swap = self.swap(repo_specific_changes_fetcher, 'get_setup_scripts_changes_status', mock_get_setup_scripts_changes_status)
        storage_models_swap = self.swap(repo_specific_changes_fetcher, 'get_changed_storage_models_filenames', mock_get_changed_storage_models_filenames)
        with versions_swap, setup_scripts_swap, storage_models_swap:
            expected_changes = ['\n### Feconf version changes:\nThis indicates that a migration may be needed\n\n', '* version_change\n', '\n### Changed setup scripts:\n', '* setup_changes\n', '\n### Changed storage models:\n', '* setup_changes\n']
            self.assertEqual(repo_specific_changes_fetcher.get_changes('release_tag'), expected_changes)

    def test_main(self) -> None:
        if False:
            return 10

        def mock_get_changes(unused_release_tag_to_diff_against: str) -> List[str]:
            if False:
                print('Hello World!')
            return ['change1', 'change2', 'change3']
        printed_lines: List[str] = []

        def mock_print(text_to_print: str) -> None:
            if False:
                return 10
            printed_lines.append(text_to_print)
        get_changes_swap = self.swap(repo_specific_changes_fetcher, 'get_changes', mock_get_changes)
        print_swap = self.swap(builtins, 'print', mock_print)
        with get_changes_swap, print_swap:
            repo_specific_changes_fetcher.main(args=['--release_tag', 'tag'])
        self.assertEqual(printed_lines, ['change1\nchange2\nchange3'])
"""Unit tests for testing import of gcloud_adapter and
install_third_party_libs. This is written separate from the main tests
for gcloud_adapter and install_third_party_libs since those tests require
importing the respective files in the start and if the files are imported
in start, then adding the same import statement in a test function
(as done in this file) creates a conflict.
"""
from __future__ import annotations
import subprocess
import sys
from core.tests import test_utils
from typing import List

class InstallThirdPartyLibsImportTests(test_utils.GenericTestBase):
    """Tests import of install third party libs."""

    def test_import_with_missing_packages(self) -> None:
        if False:
            i = 10
            return i + 15
        commands: List[List[str]] = []

        def mock_run(cmd_tokens: List[str], *_args: str, **_kwargs: str) -> None:
            if False:
                while True:
                    i = 10
            commands.append(cmd_tokens)
        run_swap = self.swap(subprocess, 'run', mock_run)
        with run_swap:
            from scripts import install_third_party_libs
        expected_commands = [[sys.executable, '-m', 'pip', 'install', version_string] for version_string in ('pip==23.1.2', 'pip-tools==6.13.0', 'setuptools==67.7.1')]
        expected_commands += [['pip-compile', '--no-emit-index-url', '--generate-hashes', 'requirements_dev.in', '--output-file', 'requirements_dev.txt'], ['pip-sync', 'requirements_dev.txt', '--pip-args', '--require-hashes --no-deps']]
        self.assertEqual(commands, expected_commands)
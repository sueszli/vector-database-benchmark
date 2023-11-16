"""Unit tests for scripts/generate_build_directory.py."""
from __future__ import annotations
from core import feconf
from core.tests import test_utils
from scripts import generate_build_directory

class Ret:
    """Return object with required attributes."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.returncode = 0

class GenerateBuildDirectoryTests(test_utils.GenericTestBase):
    """Test the methods for generate build directory."""

    def test_generate_build_dir_under_docker(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(KeyError, 'js/third_party.min.js'):
            with self.swap(feconf, 'OPPIA_IS_DOCKERIZED', True):
                generate_build_directory.main()
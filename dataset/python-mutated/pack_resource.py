from __future__ import absolute_import
import os
import inspect
from unittest2 import TestCase
__all__ = ['BasePackResourceTestCase']

class BasePackResourceTestCase(TestCase):
    """
    Base test class for all the pack resource test classes.

    Contains some utility methods for loading fixtures from disk, etc.
    """

    def get_fixture_content(self, fixture_path):
        if False:
            print('Hello World!')
        '\n        Return raw fixture content for the provided fixture path.\n\n        :param fixture_path: Fixture path relative to the tests/fixtures/ directory.\n        :type fixture_path: ``str``\n        '
        base_pack_path = self._get_base_pack_path()
        fixtures_path = os.path.join(base_pack_path, 'tests/fixtures/')
        fixture_path = os.path.join(fixtures_path, fixture_path)
        with open(fixture_path, 'r') as fp:
            content = fp.read()
        return content

    def _get_base_pack_path(self):
        if False:
            for i in range(10):
                print('nop')
        test_file_path = inspect.getfile(self.__class__)
        base_pack_path = os.path.join(os.path.dirname(test_file_path), '..')
        base_pack_path = os.path.abspath(base_pack_path)
        return base_pack_path
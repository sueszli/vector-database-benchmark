"""Tests for certbot._internal.plugins.null."""
import sys
import unittest
from unittest import mock
import pytest

class InstallerTest(unittest.TestCase):
    """Tests for certbot._internal.plugins.null.Installer."""

    def setUp(self):
        if False:
            return 10
        from certbot._internal.plugins.null import Installer
        self.installer = Installer(config=mock.MagicMock(), name='null')

    def test_it(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(self.installer.more_info(), str)
        assert [] == self.installer.get_all_names()
        assert [] == self.installer.supported_enhancements()
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
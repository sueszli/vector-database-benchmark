"""Tests for ApacheConfigurator for AugeasParserNode classes"""
import sys
import unittest
from unittest import mock
import pytest
from certbot_apache._internal.tests import util
try:
    import apacheconfig
    HAS_APACHECONFIG = True
except ImportError:
    HAS_APACHECONFIG = False

@unittest.skipIf(not HAS_APACHECONFIG, reason='Tests require apacheconfig dependency')
class ConfiguratorParserNodeTest(util.ApacheTest):
    """Test AugeasParserNode using available test configurations"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.config = util.get_apache_configurator(self.config_path, self.vhost_path, self.config_dir, self.work_dir, use_parsernode=True)
        self.vh_truth = util.get_vh_truth(self.temp_dir, 'debian_apache_2_4/multiple_vhosts')

    def test_parsernode_get_vhosts(self):
        if False:
            while True:
                i = 10
        self.config.USE_PARSERNODE = True
        vhosts = self.config.get_virtual_hosts()
        assert vhosts[0].node is not None

    def test_parsernode_get_vhosts_mismatch(self):
        if False:
            return 10
        vhosts = self.config.get_virtual_hosts_v2()
        vhosts[0].name = 'IdidntExpectThat'
        self.config.get_virtual_hosts_v2 = mock.MagicMock(return_value=vhosts)
        with pytest.raises(AssertionError):
            _ = self.config.get_virtual_hosts()
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
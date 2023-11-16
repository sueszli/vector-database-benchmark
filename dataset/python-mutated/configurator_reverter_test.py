"""Test for certbot_apache._internal.configurator implementations of reverter"""
import shutil
import sys
import unittest
from unittest import mock
import pytest
from certbot import errors
from certbot_apache._internal.tests import util

class ConfiguratorReverterTest(util.ApacheTest):
    """Test for ApacheConfigurator reverter methods"""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.config = util.get_apache_configurator(self.config_path, self.vhost_path, self.config_dir, self.work_dir)
        self.vh_truth = util.get_vh_truth(self.temp_dir, 'debian_apache_2_4/multiple_vhosts')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.config_dir)
        shutil.rmtree(self.work_dir)
        shutil.rmtree(self.temp_dir)

    def test_bad_save_checkpoint(self):
        if False:
            i = 10
            return i + 15
        self.config.reverter.add_to_checkpoint = mock.Mock(side_effect=errors.ReverterError)
        self.config.parser.add_dir(self.vh_truth[0].path, 'Test', 'bad_save_ckpt')
        with pytest.raises(errors.PluginError):
            self.config.save()

    def test_bad_save_finalize_checkpoint(self):
        if False:
            while True:
                i = 10
        self.config.reverter.finalize_checkpoint = mock.Mock(side_effect=errors.ReverterError)
        self.config.parser.add_dir(self.vh_truth[0].path, 'Test', 'bad_save_ckpt')
        with pytest.raises(errors.PluginError):
            self.config.save('Title')

    def test_finalize_save(self):
        if False:
            while True:
                i = 10
        mock_finalize = mock.Mock()
        self.config.reverter = mock_finalize
        self.config.save('Example Title')
        assert mock_finalize.is_called

    def test_revert_challenge_config(self):
        if False:
            return 10
        mock_load = mock.Mock()
        self.config.parser.aug.load = mock_load
        self.config.revert_challenge_config()
        assert mock_load.call_count == 1

    def test_revert_challenge_config_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.config.reverter.revert_temporary_config = mock.Mock(side_effect=errors.ReverterError)
        with pytest.raises(errors.PluginError):
            self.config.revert_challenge_config()

    def test_rollback_checkpoints(self):
        if False:
            return 10
        mock_load = mock.Mock()
        self.config.parser.aug.load = mock_load
        self.config.rollback_checkpoints()
        assert mock_load.call_count == 1

    def test_rollback_error(self):
        if False:
            i = 10
            return i + 15
        self.config.reverter.rollback_checkpoints = mock.Mock(side_effect=errors.ReverterError)
        with pytest.raises(errors.PluginError):
            self.config.rollback_checkpoints()

    def test_recovery_routine_reload(self):
        if False:
            while True:
                i = 10
        mock_load = mock.Mock()
        self.config.parser.aug.load = mock_load
        self.config.recovery_routine()
        assert mock_load.call_count == 1
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
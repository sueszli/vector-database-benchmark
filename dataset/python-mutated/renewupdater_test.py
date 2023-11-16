"""Tests for renewal updater interfaces"""
import sys
import unittest
from unittest import mock
import pytest
from certbot import interfaces
from certbot._internal import main
from certbot._internal import updater
from certbot.plugins import enhancements
import certbot.tests.util as test_util

class RenewUpdaterTest(test_util.ConfigTestCase):
    """Tests for interfaces.RenewDeployer and interfaces.GenericUpdater"""

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.generic_updater = mock.MagicMock(spec=interfaces.GenericUpdater)
        self.generic_updater.restart = mock.MagicMock()
        self.renew_deployer = mock.MagicMock(spec=interfaces.RenewDeployer)
        self.mockinstaller = mock.MagicMock(spec=enhancements.AutoHSTSEnhancement)

    @mock.patch('certbot._internal.main._get_and_save_cert')
    @mock.patch('certbot._internal.plugins.selection.choose_configurator_plugins')
    @mock.patch('certbot._internal.plugins.selection.get_unprepared_installer')
    @test_util.patch_display_util()
    def test_server_updates(self, _, mock_geti, mock_select, mock_getsave):
        if False:
            while True:
                i = 10
        mock_getsave.return_value = mock.MagicMock()
        mock_generic_updater = self.generic_updater
        mock_select.return_value = (mock_generic_updater, None)
        mock_geti.return_value = mock_generic_updater
        with mock.patch('certbot._internal.main._init_le_client'):
            main.renew_cert(self.config, None, mock.MagicMock())
        assert mock_generic_updater.restart.called
        mock_generic_updater.restart.reset_mock()
        mock_generic_updater.generic_updates.reset_mock()
        updater.run_generic_updaters(self.config, mock.MagicMock(), None)
        assert mock_generic_updater.generic_updates.call_count == 1
        assert mock_generic_updater.restart.called is False

    def test_renew_deployer(self):
        if False:
            return 10
        lineage = mock.MagicMock()
        mock_deployer = self.renew_deployer
        updater.run_renewal_deployer(self.config, lineage, mock_deployer)
        mock_deployer.renew_deploy.assert_called_with(lineage)

    @mock.patch('certbot._internal.updater.logger.debug')
    def test_updater_skip_dry_run(self, mock_log):
        if False:
            for i in range(10):
                print('nop')
        self.config.dry_run = True
        updater.run_generic_updaters(self.config, None, None)
        assert mock_log.called
        assert mock_log.call_args[0][0] == 'Skipping updaters in dry-run mode.'

    @mock.patch('certbot._internal.updater.logger.debug')
    def test_deployer_skip_dry_run(self, mock_log):
        if False:
            return 10
        self.config.dry_run = True
        updater.run_renewal_deployer(self.config, None, None)
        assert mock_log.called
        assert mock_log.call_args[0][0] == 'Skipping renewal deployer in dry-run mode.'

    @mock.patch('certbot._internal.plugins.selection.get_unprepared_installer')
    def test_enhancement_updates(self, mock_geti):
        if False:
            i = 10
            return i + 15
        mock_geti.return_value = self.mockinstaller
        updater.run_generic_updaters(self.config, mock.MagicMock(), None)
        assert self.mockinstaller.update_autohsts.called
        assert self.mockinstaller.update_autohsts.call_count == 1

    def test_enhancement_deployer(self):
        if False:
            return 10
        updater.run_renewal_deployer(self.config, mock.MagicMock(), self.mockinstaller)
        assert self.mockinstaller.deploy_autohsts.called

    @mock.patch('certbot._internal.plugins.selection.get_unprepared_installer')
    def test_enhancement_updates_not_called(self, mock_geti):
        if False:
            while True:
                i = 10
        self.config.disable_renew_updates = True
        mock_geti.return_value = self.mockinstaller
        updater.run_generic_updaters(self.config, mock.MagicMock(), None)
        assert self.mockinstaller.update_autohsts.called is False

    def test_enhancement_deployer_not_called(self):
        if False:
            for i in range(10):
                print('nop')
        self.config.disable_renew_updates = True
        updater.run_renewal_deployer(self.config, mock.MagicMock(), self.mockinstaller)
        assert self.mockinstaller.deploy_autohsts.called is False

    @mock.patch('certbot._internal.plugins.selection.get_unprepared_installer')
    def test_enhancement_no_updater(self, mock_geti):
        if False:
            while True:
                i = 10
        FAKEINDEX = [{'name': 'Test', 'class': enhancements.AutoHSTSEnhancement, 'updater_function': None, 'deployer_function': 'deploy_autohsts', 'enable_function': 'enable_autohsts'}]
        mock_geti.return_value = self.mockinstaller
        with mock.patch('certbot.plugins.enhancements._INDEX', FAKEINDEX):
            updater.run_generic_updaters(self.config, mock.MagicMock(), None)
        assert self.mockinstaller.update_autohsts.called is False

    def test_enhancement_no_deployer(self):
        if False:
            while True:
                i = 10
        FAKEINDEX = [{'name': 'Test', 'class': enhancements.AutoHSTSEnhancement, 'updater_function': 'deploy_autohsts', 'deployer_function': None, 'enable_function': 'enable_autohsts'}]
        with mock.patch('certbot.plugins.enhancements._INDEX', FAKEINDEX):
            updater.run_renewal_deployer(self.config, mock.MagicMock(), self.mockinstaller)
        assert self.mockinstaller.deploy_autohsts.called is False
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
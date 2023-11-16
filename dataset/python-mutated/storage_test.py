"""Tests for certbot.plugins.storage.PluginStorage"""
import json
import sys
from typing import Iterable
from typing import List
from typing import Optional
import unittest
from unittest import mock
import pytest
from certbot import errors
from certbot.compat import filesystem
from certbot.compat import os
from certbot.tests import util as test_util

class PluginStorageTest(test_util.ConfigTestCase):
    """Test for certbot.plugins.storage.PluginStorage"""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.plugin_cls = test_util.DummyInstaller
        filesystem.mkdir(self.config.config_dir)
        with mock.patch('certbot.reverter.util'):
            self.plugin = self.plugin_cls(config=self.config, name='mockplugin')

    def test_load_errors_cant_read(self):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(self.config.config_dir, '.pluginstorage.json'), 'w') as fh:
            fh.write('dummy')
        mock_open = mock.mock_open()
        mock_open.side_effect = IOError
        self.plugin.storage._storagepath = os.path.join(self.config.config_dir, '.pluginstorage.json')
        with mock.patch('builtins.open', mock_open):
            with mock.patch('certbot.compat.os.path.isfile', return_value=True):
                with mock.patch('certbot.reverter.util'):
                    with pytest.raises(errors.PluginStorageError):
                        self.plugin.storage._load()

    def test_load_errors_empty(self):
        if False:
            i = 10
            return i + 15
        with open(os.path.join(self.config.config_dir, '.pluginstorage.json'), 'w') as fh:
            fh.write('')
        with mock.patch('certbot.plugins.storage.logger.debug') as mock_log:
            with mock.patch('certbot.reverter.util'):
                nocontent = self.plugin_cls(self.config, 'mockplugin')
            with pytest.raises(KeyError):
                nocontent.storage.fetch('value')
            assert mock_log.called
            assert 'no values loaded' in mock_log.call_args[0][0]

    def test_load_errors_corrupted(self):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(self.config.config_dir, '.pluginstorage.json'), 'w') as fh:
            fh.write('invalid json')
        with mock.patch('certbot.plugins.storage.logger.error') as mock_log:
            with mock.patch('certbot.reverter.util'):
                corrupted = self.plugin_cls(self.config, 'mockplugin')
            with pytest.raises(errors.PluginError):
                corrupted.storage.fetch('value')
            assert 'is corrupted' in mock_log.call_args[0][0]

    def test_save_errors_cant_serialize(self):
        if False:
            return 10
        with mock.patch('certbot.plugins.storage.logger.error') as mock_log:
            self.plugin.storage._initialized = True
            self.plugin.storage._storagepath = '/tmp/whatever'
            self.plugin.storage._data = self.plugin_cls
            with pytest.raises(errors.PluginStorageError):
                self.plugin.storage.save()
            assert 'Could not serialize' in mock_log.call_args[0][0]

    def test_save_errors_unable_to_write_file(self):
        if False:
            for i in range(10):
                print('nop')
        mock_open = mock.mock_open()
        mock_open.side_effect = IOError
        with mock.patch('certbot.compat.filesystem.open', mock_open):
            with mock.patch('certbot.plugins.storage.logger.error') as mock_log:
                self.plugin.storage._data = {'valid': 'data'}
                self.plugin.storage._initialized = True
                self.plugin.storage._storagepath = '/tmp/whatever'
                with pytest.raises(errors.PluginStorageError):
                    self.plugin.storage.save()
                assert 'Could not write' in mock_log.call_args[0][0]

    def test_save_uninitialized(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('certbot.reverter.util'):
            with pytest.raises(errors.PluginStorageError):
                self.plugin_cls(self.config, 'x').storage.save()

    def test_namespace_isolation(self):
        if False:
            return 10
        with mock.patch('certbot.reverter.util'):
            plugin1 = self.plugin_cls(self.config, 'first')
            plugin2 = self.plugin_cls(self.config, 'second')
        plugin1.storage.put('first_key', 'first_value')
        with pytest.raises(KeyError):
            plugin2.storage.fetch('first_key')
        with pytest.raises(KeyError):
            plugin2.storage.fetch('first')
        assert plugin1.storage.fetch('first_key') == 'first_value'

    def test_saved_state(self):
        if False:
            print('Hello World!')
        self.plugin.storage.put('testkey', 'testvalue')
        self.plugin.storage.save()
        with mock.patch('certbot.reverter.util'):
            another = self.plugin_cls(self.config, 'mockplugin')
        assert another.storage.fetch('testkey') == 'testvalue'
        with open(os.path.join(self.config.config_dir, '.pluginstorage.json'), 'r') as fh:
            psdata = fh.read()
        psjson = json.loads(psdata)
        assert 'mockplugin' in psjson.keys()
        assert len(psjson) == 1
        assert psjson['mockplugin']['testkey'] == 'testvalue'
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
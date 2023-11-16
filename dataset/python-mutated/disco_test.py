"""Tests for certbot._internal.plugins.disco."""
import functools
import string
import sys
from typing import List
import unittest
from unittest import mock
import pytest
from certbot import errors
from certbot import interfaces
from certbot._internal.plugins import null
from certbot._internal.plugins import standalone
from certbot._internal.plugins import webroot
if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

class _EntryPointLoadFail(importlib_metadata.EntryPoint):

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Loading failure')
EP_SA = importlib_metadata.EntryPoint(name='sa', value='certbot._internal.plugins.standalone:Authenticator', group='certbot.plugins')
EP_WR = importlib_metadata.EntryPoint(name='wr', value='certbot._internal.plugins.webroot:Authenticator', group='certbot.plugins')
EP_SA_LOADFAIL = _EntryPointLoadFail(name='sa', value='certbot._internal.plugins.standalone:Authenticator', group='certbot.plugins')

class PluginEntryPointTest(unittest.TestCase):
    """Tests for certbot._internal.plugins.disco.PluginEntryPoint."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.ep1 = importlib_metadata.EntryPoint(name='ep1', value='p1.ep1:Authenticator', group='certbot.plugins')
        self.ep1prim = importlib_metadata.EntryPoint(name='ep1', value='p2.pe2:Authenticator', group='certbot.plugins')
        self.ep2 = importlib_metadata.EntryPoint(name='ep2', value='p2.foo.ep2:Authenticator', group='certbot.plugins')
        self.ep3 = importlib_metadata.EntryPoint(name='ep3', value='a.ep3:Authenticator', group='certbot.plugins')
        from certbot._internal.plugins.disco import PluginEntryPoint
        self.plugin_ep = PluginEntryPoint(EP_SA)

    def test_entry_point_to_plugin_name_not_prefixed(self):
        if False:
            i = 10
            return i + 15
        from certbot._internal.plugins.disco import PluginEntryPoint
        names = {self.ep1: 'ep1', self.ep1prim: 'ep1', self.ep2: 'ep2', self.ep3: 'ep3', EP_SA: 'sa'}
        for (entry_point, name) in names.items():
            assert name == PluginEntryPoint.entry_point_to_plugin_name(entry_point)

    def test_description(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'server locally' in self.plugin_ep.description

    def test_description_with_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.plugin_ep.plugin_cls = mock.MagicMock(description='Desc')
        assert 'Desc (sa)' == self.plugin_ep.description_with_name

    def test_long_description(self):
        if False:
            return 10
        self.plugin_ep.plugin_cls = mock.MagicMock(long_description='Long desc')
        assert 'Long desc' == self.plugin_ep.long_description

    def test_long_description_nonexistent(self):
        if False:
            print('Hello World!')
        self.plugin_ep.plugin_cls = mock.MagicMock(description='Long desc not found', spec=['description'])
        assert 'Long desc not found' == self.plugin_ep.long_description

    def test_ifaces(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.plugin_ep.ifaces((interfaces.Authenticator,))
        assert not self.plugin_ep.ifaces((interfaces.Installer,))
        assert not self.plugin_ep.ifaces((interfaces.Installer, interfaces.Authenticator))

    def test__init__(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.plugin_ep.initialized is False
        assert self.plugin_ep.prepared is False
        assert self.plugin_ep.misconfigured is False
        assert self.plugin_ep.available is False
        assert self.plugin_ep.problem is None
        assert self.plugin_ep.entry_point is EP_SA
        assert 'sa' == self.plugin_ep.name
        assert self.plugin_ep.plugin_cls is standalone.Authenticator

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        config = mock.MagicMock()
        plugin = self.plugin_ep.init(config=config)
        assert self.plugin_ep.initialized is True
        assert plugin.config is config
        assert self.plugin_ep.init() is plugin
        assert plugin.config is config
        assert self.plugin_ep.init(123) is plugin
        assert plugin.config is config
        assert self.plugin_ep.prepared is False
        assert self.plugin_ep.misconfigured is False
        assert self.plugin_ep.available is False

    def test_prepare(self):
        if False:
            i = 10
            return i + 15
        config = mock.MagicMock()
        self.plugin_ep.init(config=config)
        self.plugin_ep.prepare()
        assert self.plugin_ep.prepared
        assert self.plugin_ep.misconfigured is False
        str(self.plugin_ep)

    def test_prepare_misconfigured(self):
        if False:
            print('Hello World!')
        plugin = mock.MagicMock()
        plugin.prepare.side_effect = errors.MisconfigurationError
        self.plugin_ep._initialized = plugin
        assert isinstance(self.plugin_ep.prepare(), errors.MisconfigurationError)
        assert self.plugin_ep.prepared
        assert self.plugin_ep.misconfigured
        assert isinstance(self.plugin_ep.problem, errors.MisconfigurationError)
        assert self.plugin_ep.available

    def test_prepare_no_installation(self):
        if False:
            print('Hello World!')
        plugin = mock.MagicMock()
        plugin.prepare.side_effect = errors.NoInstallationError
        self.plugin_ep._initialized = plugin
        assert isinstance(self.plugin_ep.prepare(), errors.NoInstallationError)
        assert self.plugin_ep.prepared is True
        assert self.plugin_ep.misconfigured is False
        assert self.plugin_ep.available is False

    def test_prepare_generic_plugin_error(self):
        if False:
            return 10
        plugin = mock.MagicMock()
        plugin.prepare.side_effect = errors.PluginError
        self.plugin_ep._initialized = plugin
        assert isinstance(self.plugin_ep.prepare(), errors.PluginError)
        assert self.plugin_ep.prepared
        assert self.plugin_ep.misconfigured is False
        assert self.plugin_ep.available is False

    def test_str(self):
        if False:
            print('Hello World!')
        output = str(self.plugin_ep)
        assert 'Authenticator' in output
        assert 'Installer' not in output
        assert 'Plugin' in output

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'PluginEntryPoint#sa' == repr(self.plugin_ep)

class PluginsRegistryTest(unittest.TestCase):
    """Tests for certbot._internal.plugins.disco.PluginsRegistry."""

    @classmethod
    def _create_new_registry(cls, plugins):
        if False:
            return 10
        from certbot._internal.plugins.disco import PluginsRegistry
        return PluginsRegistry(plugins)

    def setUp(self):
        if False:
            return 10
        self.plugin_ep = mock.MagicMock()
        self.plugin_ep.name = 'mock'
        self.plugin_ep.__hash__.side_effect = TypeError
        self.plugins = {self.plugin_ep.name: self.plugin_ep}
        self.reg = self._create_new_registry(self.plugins)
        self.ep1 = importlib_metadata.EntryPoint(name='ep1', value='p1.ep1', group='certbot.plugins')

    def test_find_all(self):
        if False:
            i = 10
            return i + 15
        from certbot._internal.plugins.disco import PluginsRegistry
        with mock.patch('certbot._internal.plugins.disco.importlib_metadata') as mock_meta:
            mock_meta.entry_points.side_effect = [[EP_SA], [EP_WR, self.ep1]]
            with mock.patch.object(importlib_metadata.EntryPoint, 'load') as mock_load:
                mock_load.side_effect = [standalone.Authenticator, webroot.Authenticator, null.Installer, null.Installer]
                plugins = PluginsRegistry.find_all()
        assert plugins['sa'].plugin_cls is standalone.Authenticator
        assert plugins['sa'].entry_point is EP_SA
        assert plugins['wr'].plugin_cls is webroot.Authenticator
        assert plugins['wr'].entry_point is EP_WR
        assert plugins['ep1'].plugin_cls is null.Installer
        assert plugins['ep1'].entry_point is self.ep1
        assert 'p1:ep1' not in plugins

    def test_find_all_error_message(self):
        if False:
            for i in range(10):
                print('nop')
        from certbot._internal.plugins.disco import PluginsRegistry
        with mock.patch('certbot._internal.plugins.disco.importlib_metadata') as mock_meta:
            mock_meta.entry_points.side_effect = [[EP_SA_LOADFAIL], [EP_WR, self.ep1]]
            with self.assertRaises(errors.PluginError) as cm:
                PluginsRegistry.find_all()
            assert "standalone' plugin errored" in str(cm.exception)

    def test_getitem(self):
        if False:
            i = 10
            return i + 15
        assert self.plugin_ep == self.reg['mock']

    def test_iter(self):
        if False:
            i = 10
            return i + 15
        assert ['mock'] == list(self.reg)

    def test_len(self):
        if False:
            for i in range(10):
                print('nop')
        assert 0 == len(self._create_new_registry({}))
        assert 1 == len(self.reg)

    def test_init(self):
        if False:
            return 10
        self.plugin_ep.init.return_value = 'baz'
        assert ['baz'] == self.reg.init('bar')
        self.plugin_ep.init.assert_called_once_with('bar')

    def test_filter(self):
        if False:
            i = 10
            return i + 15
        assert self.plugins == self.reg.filter(lambda p_ep: p_ep.name.startswith('m'))
        assert {} == self.reg.filter(lambda p_ep: p_ep.name.startswith('b'))

    def test_ifaces(self):
        if False:
            return 10
        self.plugin_ep.ifaces.return_value = True
        assert self.plugins == self.reg.ifaces()._plugins
        self.plugin_ep.ifaces.return_value = False
        assert {} == self.reg.ifaces()._plugins

    def test_prepare(self):
        if False:
            i = 10
            return i + 15
        self.plugin_ep.prepare.return_value = 'baz'
        assert ['baz'] == self.reg.prepare()
        self.plugin_ep.prepare.assert_called_once_with()

    def test_prepare_order(self):
        if False:
            while True:
                i = 10
        order: List[str] = []
        plugins = {c: mock.MagicMock(prepare=functools.partial(order.append, c)) for c in string.ascii_letters}
        reg = self._create_new_registry(plugins)
        reg.prepare()
        assert order == sorted(string.ascii_letters)

    def test_available(self):
        if False:
            for i in range(10):
                print('nop')
        self.plugin_ep.available = True
        assert self.plugins == self.reg.available()._plugins
        self.plugin_ep.available = False
        assert {} == self.reg.available()._plugins

    def test_find_init(self):
        if False:
            while True:
                i = 10
        assert self.reg.find_init(mock.Mock()) is None
        self.plugin_ep.initialized = True
        assert self.reg.find_init(self.plugin_ep.init()) is self.plugin_ep

    def test_repr(self):
        if False:
            print('Hello World!')
        self.plugin_ep.__repr__ = lambda _: 'PluginEntryPoint#mock'
        assert 'PluginsRegistry(PluginEntryPoint#mock)' == repr(self.reg)

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'No plugins' == str(self._create_new_registry({}))
        self.plugin_ep.__str__ = lambda _: 'Mock'
        assert 'Mock' == str(self.reg)
        plugins = {self.plugin_ep.name: self.plugin_ep, 'foo': 'Bar'}
        reg = self._create_new_registry(plugins)
        assert 'Bar\n\nMock' == str(reg)
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))
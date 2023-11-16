from __future__ import annotations
import logging
import re
import sys
from unittest.mock import patch
import pytest
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from flask_babel import lazy_gettext
from wtforms import BooleanField, Field, StringField
from airflow.exceptions import AirflowOptionalProviderFeatureException
from airflow.providers_manager import HookClassProvider, LazyDictWithCache, PluginInfo, ProviderInfo, ProvidersManager

class TestProviderManager:

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        if False:
            i = 10
            return i + 15
        self._caplog = caplog

    @pytest.fixture(autouse=True, scope='function')
    def clean(self):
        if False:
            i = 10
            return i + 15
        'The tests depend on a clean state of a ProvidersManager.'
        ProvidersManager().__init__()

    def test_providers_are_loaded(self):
        if False:
            for i in range(10):
                print('nop')
        with self._caplog.at_level(logging.WARNING):
            provider_manager = ProvidersManager()
            provider_list = list(provider_manager.providers.keys())
            for provider in provider_list:
                package_name = provider_manager.providers[provider].data['package-name']
                version = provider_manager.providers[provider].version
                assert re.search('[0-9]*\\.[0-9]*\\.[0-9]*.*', version)
                assert package_name == provider
            assert len(provider_list) > 65
            assert [] == self._caplog.records

    def test_hooks_deprecation_warnings_generated(self):
        if False:
            print('Hello World!')
        with pytest.warns(expected_warning=DeprecationWarning, match='hook-class-names') as warning_records:
            providers_manager = ProvidersManager()
            providers_manager._provider_dict['test-package'] = ProviderInfo(version='0.0.1', data={'hook-class-names': ['airflow.providers.sftp.hooks.sftp.SFTPHook']}, package_or_source='package')
            providers_manager._discover_hooks()
        assert warning_records

    def test_hooks_deprecation_warnings_not_generated(self):
        if False:
            return 10
        with pytest.warns(expected_warning=None) as warning_records:
            providers_manager = ProvidersManager()
            providers_manager._provider_dict['apache-airflow-providers-sftp'] = ProviderInfo(version='0.0.1', data={'hook-class-names': ['airflow.providers.sftp.hooks.sftp.SFTPHook'], 'connection-types': [{'hook-class-name': 'airflow.providers.sftp.hooks.sftp.SFTPHook', 'connection-type': 'sftp'}]}, package_or_source='package')
            providers_manager._discover_hooks()
        assert [] == [w.message for w in warning_records.list if 'hook-class-names' in str(w.message)]

    def test_warning_logs_generated(self):
        if False:
            return 10
        with self._caplog.at_level(logging.WARNING):
            providers_manager = ProvidersManager()
            providers_manager._provider_dict['apache-airflow-providers-sftp'] = ProviderInfo(version='0.0.1', data={'hook-class-names': ['airflow.providers.sftp.hooks.sftp.SFTPHook'], 'connection-types': [{'hook-class-name': 'airflow.providers.sftp.hooks.sftp.SFTPHook', 'connection-type': 'wrong-connection-type'}]}, package_or_source='package')
            providers_manager._discover_hooks()
            _ = providers_manager._hooks_lazy_dict['wrong-connection-type']
        assert len(self._caplog.records) == 1
        assert 'Inconsistency!' in self._caplog.records[0].message
        assert 'sftp' not in providers_manager.hooks

    def test_warning_logs_not_generated(self):
        if False:
            for i in range(10):
                print('nop')
        with self._caplog.at_level(logging.WARNING):
            providers_manager = ProvidersManager()
            providers_manager._provider_dict['apache-airflow-providers-sftp'] = ProviderInfo(version='0.0.1', data={'hook-class-names': ['airflow.providers.sftp.hooks.sftp.SFTPHook'], 'connection-types': [{'hook-class-name': 'airflow.providers.sftp.hooks.sftp.SFTPHook', 'connection-type': 'sftp'}]}, package_or_source='package')
            providers_manager._discover_hooks()
            _ = providers_manager._hooks_lazy_dict['sftp']
        assert not self._caplog.records
        assert 'sftp' in providers_manager.hooks

    def test_already_registered_conn_type_in_provide(self):
        if False:
            while True:
                i = 10
        with self._caplog.at_level(logging.WARNING):
            providers_manager = ProvidersManager()
            providers_manager._provider_dict['apache-airflow-providers-dummy'] = ProviderInfo(version='0.0.1', data={'connection-types': [{'hook-class-name': 'airflow.providers.dummy.hooks.dummy.DummyHook', 'connection-type': 'dummy'}, {'hook-class-name': 'airflow.providers.dummy.hooks.dummy.DummyHook2', 'connection-type': 'dummy'}]}, package_or_source='package')
            providers_manager._discover_hooks()
            _ = providers_manager._hooks_lazy_dict['dummy']
        assert len(self._caplog.records) == 1
        assert "The connection type 'dummy' is already registered" in self._caplog.records[0].message
        assert "different class names: 'airflow.providers.dummy.hooks.dummy.DummyHook' and 'airflow.providers.dummy.hooks.dummy.DummyHook2'." in self._caplog.records[0].message

    def test_providers_manager_register_plugins(self):
        if False:
            while True:
                i = 10
        providers_manager = ProvidersManager()
        providers_manager._provider_dict['apache-airflow-providers-apache-hive'] = ProviderInfo(version='0.0.1', data={'plugins': [{'name': 'plugin1', 'plugin-class': 'airflow.providers.apache.hive.plugins.hive.HivePlugin'}]}, package_or_source='package')
        providers_manager._discover_plugins()
        assert len(providers_manager._plugins_set) == 1
        assert providers_manager._plugins_set.pop() == PluginInfo(name='plugin1', plugin_class='airflow.providers.apache.hive.plugins.hive.HivePlugin', provider_name='apache-airflow-providers-apache-hive')

    def test_hooks(self):
        if False:
            return 10
        with pytest.warns(expected_warning=None) as warning_records:
            with self._caplog.at_level(logging.WARNING):
                provider_manager = ProvidersManager()
                connections_list = list(provider_manager.hooks.keys())
                assert len(connections_list) > 60
        if len(self._caplog.records) != 0:
            for record in self._caplog.records:
                print(record.message, file=sys.stderr)
                print(record.exc_info, file=sys.stderr)
            raise AssertionError('There are warnings generated during hook imports. Please fix them')
        assert [] == [w.message for w in warning_records.list if 'hook-class-names' in str(w.message)]

    @pytest.mark.execution_timeout(150)
    def test_hook_values(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(expected_warning=None) as warning_records:
            with self._caplog.at_level(logging.WARNING):
                provider_manager = ProvidersManager()
                connections_list = list(provider_manager.hooks.values())
                assert len(connections_list) > 60
        if len(self._caplog.records) != 0:
            for record in self._caplog.records:
                print(record.message, file=sys.stderr)
                print(record.exc_info, file=sys.stderr)
            raise AssertionError('There are warnings generated during hook imports. Please fix them')
        assert [] == [w.message for w in warning_records.list if 'hook-class-names' in str(w.message)]

    def test_connection_form_widgets(self):
        if False:
            while True:
                i = 10
        provider_manager = ProvidersManager()
        connections_form_widgets = list(provider_manager.connection_form_widgets.keys())
        assert len(connections_form_widgets) > 29

    @pytest.mark.parametrize('scenario', ['prefix', 'no_prefix', 'both_1', 'both_2'])
    def test_connection_form__add_widgets_prefix_backcompat(self, scenario):
        if False:
            i = 10
            return i + 15
        "\n        When the field name is prefixed, it should be used as is.\n        When not prefixed, we should add the prefix\n        When there's a collision, the one that appears first in the list will be used.\n        "

        class MyHook:
            conn_type = 'test'
        provider_manager = ProvidersManager()
        widget_field = StringField(lazy_gettext('My Param'), widget=BS3TextFieldWidget())
        dummy_field = BooleanField(label=lazy_gettext('Dummy param'), description='dummy')
        widgets: dict[str, Field] = {}
        if scenario == 'prefix':
            widgets['extra__test__my_param'] = widget_field
        elif scenario == 'no_prefix':
            widgets['my_param'] = widget_field
        elif scenario == 'both_1':
            widgets['my_param'] = widget_field
            widgets['extra__test__my_param'] = dummy_field
        elif scenario == 'both_2':
            widgets['extra__test__my_param'] = widget_field
            widgets['my_param'] = dummy_field
        else:
            raise Exception('unexpected')
        provider_manager._add_widgets(package_name='abc', hook_class=MyHook, widgets=widgets)
        assert provider_manager.connection_form_widgets['extra__test__my_param'].field == widget_field

    def test_connection_field_behaviors_placeholders_prefix(self):
        if False:
            return 10

        class MyHook:
            conn_type = 'test'

            @classmethod
            def get_ui_field_behaviour(cls):
                if False:
                    return 10
                return {'hidden_fields': ['host', 'schema'], 'relabeling': {}, 'placeholders': {'abc': 'hi', 'extra__anything': 'n/a', 'password': 'blah'}}
        provider_manager = ProvidersManager()
        provider_manager._add_customized_fields(package_name='abc', hook_class=MyHook, customized_fields=MyHook.get_ui_field_behaviour())
        expected = {'extra__test__abc': 'hi', 'extra__anything': 'n/a', 'password': 'blah'}
        assert provider_manager.field_behaviours['test']['placeholders'] == expected

    def test_connection_form_widgets_fields_order(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that order of connection for widgets preserved by original Hook order.'
        test_conn_type = 'test'
        field_prefix = f'extra__{test_conn_type}__'
        field_names = ('yyy_param', 'aaa_param', '000_param', 'foo', 'bar', 'spam', 'egg')
        expected_field_names_order = tuple((f'{field_prefix}{f}' for f in field_names))

        class TestHook:
            conn_type = test_conn_type
        provider_manager = ProvidersManager()
        provider_manager._connection_form_widgets = {}
        provider_manager._add_widgets(package_name='mock', hook_class=TestHook, widgets={f: BooleanField(lazy_gettext('Dummy param')) for f in expected_field_names_order})
        actual_field_names_order = tuple((key for key in provider_manager.connection_form_widgets.keys() if key.startswith(field_prefix)))
        assert actual_field_names_order == expected_field_names_order, 'Not keeping original fields order'

    def test_connection_form_widgets_fields_order_multiple_hooks(self):
        if False:
            while True:
                i = 10
        '\n        Check that order of connection for widgets preserved by original Hooks order.\n        Even if different hooks specified field with the same connection type.\n        '
        test_conn_type = 'test'
        field_prefix = f'extra__{test_conn_type}__'
        field_names_hook_1 = ('foo', 'bar', 'spam', 'egg')
        field_names_hook_2 = ('yyy_param', 'aaa_param', '000_param')
        expected_field_names_order = tuple((f'{field_prefix}{f}' for f in [*field_names_hook_1, *field_names_hook_2]))

        class TestHook1:
            conn_type = test_conn_type

        class TestHook2:
            conn_type = 'another'
        provider_manager = ProvidersManager()
        provider_manager._connection_form_widgets = {}
        provider_manager._add_widgets(package_name='mock', hook_class=TestHook1, widgets={f'{field_prefix}{f}': BooleanField(lazy_gettext('Dummy param')) for f in field_names_hook_1})
        provider_manager._add_widgets(package_name='another_mock', hook_class=TestHook2, widgets={f'{field_prefix}{f}': BooleanField(lazy_gettext('Dummy param')) for f in field_names_hook_2})
        actual_field_names_order = tuple((key for key in provider_manager.connection_form_widgets.keys() if key.startswith(field_prefix)))
        assert actual_field_names_order == expected_field_names_order, 'Not keeping original fields order'

    def test_field_behaviours(self):
        if False:
            i = 10
            return i + 15
        provider_manager = ProvidersManager()
        connections_with_field_behaviours = list(provider_manager.field_behaviours.keys())
        assert len(connections_with_field_behaviours) > 16

    def test_extra_links(self):
        if False:
            return 10
        provider_manager = ProvidersManager()
        extra_link_class_names = list(provider_manager.extra_links_class_names)
        assert len(extra_link_class_names) > 6

    def test_logging(self):
        if False:
            print('Hello World!')
        provider_manager = ProvidersManager()
        logging_class_names = list(provider_manager.logging_class_names)
        assert len(logging_class_names) > 5

    def test_secrets_backends(self):
        if False:
            print('Hello World!')
        provider_manager = ProvidersManager()
        secrets_backends_class_names = list(provider_manager.secrets_backend_class_names)
        assert len(secrets_backends_class_names) > 4

    def test_auth_backends(self):
        if False:
            i = 10
            return i + 15
        provider_manager = ProvidersManager()
        auth_backend_module_names = list(provider_manager.auth_backend_module_names)
        assert len(auth_backend_module_names) > 0

    def test_trigger(self):
        if False:
            print('Hello World!')
        provider_manager = ProvidersManager()
        trigger_class_names = list(provider_manager.trigger)
        assert len(trigger_class_names) > 10

    def test_notification(self):
        if False:
            return 10
        provider_manager = ProvidersManager()
        notification_class_names = list(provider_manager.notification)
        assert len(notification_class_names) > 5

    @patch('airflow.providers_manager.import_string')
    def test_optional_feature_no_warning(self, mock_importlib_import_string):
        if False:
            for i in range(10):
                print('nop')
        with self._caplog.at_level(logging.WARNING):
            mock_importlib_import_string.side_effect = AirflowOptionalProviderFeatureException()
            providers_manager = ProvidersManager()
            providers_manager._hook_provider_dict['test_connection'] = HookClassProvider(package_name='test_package', hook_class_name='HookClass')
            providers_manager._import_hook(hook_class_name=None, provider_info=None, package_name=None, connection_type='test_connection')
            assert [] == self._caplog.messages

    @patch('airflow.providers_manager.import_string')
    def test_optional_feature_debug(self, mock_importlib_import_string):
        if False:
            i = 10
            return i + 15
        with self._caplog.at_level(logging.INFO):
            mock_importlib_import_string.side_effect = AirflowOptionalProviderFeatureException()
            providers_manager = ProvidersManager()
            providers_manager._hook_provider_dict['test_connection'] = HookClassProvider(package_name='test_package', hook_class_name='HookClass')
            providers_manager._import_hook(hook_class_name=None, provider_info=None, package_name=None, connection_type='test_connection')
            assert ["Optional provider feature disabled when importing 'HookClass' from 'test_package' package"] == self._caplog.messages

@pytest.mark.parametrize('value, expected_outputs,', [('a', 'a'), (1, 1), (None, None), (lambda : 0, 0), (lambda : None, None), (lambda : 'z', 'z')])
def test_lazy_cache_dict_resolving(value, expected_outputs):
    if False:
        print('Hello World!')
    lazy_cache_dict = LazyDictWithCache()
    lazy_cache_dict['key'] = value
    assert lazy_cache_dict['key'] == expected_outputs
    assert lazy_cache_dict['key'] == expected_outputs

def test_lazy_cache_dict_raises_error():
    if False:
        print('Hello World!')

    def raise_method():
        if False:
            print('Hello World!')
        raise Exception('test')
    lazy_cache_dict = LazyDictWithCache()
    lazy_cache_dict['key'] = raise_method
    with pytest.raises(Exception, match='test'):
        _ = lazy_cache_dict['key']
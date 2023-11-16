import pytest
from ckan.common import config
import ckan.lib.app_globals as app_globals
import ckan.model as model
import ckan.logic as logic
import ckan.tests.helpers as helpers

class TestConfigOptionUpdatePluginNotEnabled(object):

    def test_updating_unregistered_core_setting_not_allowed(self):
        if False:
            print('Hello World!')
        key = 'ckan.datasets_per_page'
        value = 5
        params = {key: value}
        with pytest.raises(logic.ValidationError):
            helpers.call_action('config_option_update', **params)

    def test_updating_unregistered_new_setting_not_allowed(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'ckanext.example_iconfigurer.test_conf'
        value = 'Test value'
        params = {key: value}
        with pytest.raises(logic.ValidationError):
            helpers.call_action('config_option_update', **params)

@pytest.mark.ckan_config('ckan.plugins', u'example_iconfigurer')
@pytest.mark.usefixtures('clean_db', 'with_plugins')
class TestConfigOptionUpdatePluginEnabled(object):

    def test_update_registered_core_value(self, ckan_config):
        if False:
            for i in range(10):
                print('nop')
        key = 'ckan.datasets_per_page'
        value = 5
        params = {key: value}
        new_config = helpers.call_action('config_option_update', **params)
        assert new_config[key] == value
        assert ckan_config[key] == value
        globals_key = app_globals.get_globals_key(key)
        assert hasattr(app_globals.app_globals, globals_key)
        obj = model.Session.query(model.SystemInfo).filter_by(key=key).first()
        assert obj.value == str(value)

    def test_update_registered_external_value(self):
        if False:
            while True:
                i = 10
        key = 'ckanext.example_iconfigurer.test_conf'
        value = 'Test value'
        params = {key: value}
        assert not config.get(key)
        new_config = helpers.call_action('config_option_update', **params)
        assert new_config[key] == value
        assert config[key] == value
        obj = model.Session.query(model.SystemInfo).filter_by(key=key).first()
        assert obj.value == value
        globals_key = app_globals.get_globals_key(key)
        assert not getattr(app_globals.app_globals, globals_key, None)

    def test_update_registered_core_value_in_list(self):
        if False:
            return 10
        'Registering a core key/value will allow it to be included in the\n        list returned by config_option_list action.'
        key = 'ckan.datasets_per_page'
        value = 5
        params = {key: value}
        helpers.call_action('config_option_update', **params)
        option_list = helpers.call_action('config_option_list')
        assert key in option_list

    def test_update_registered_core_value_in_show(self):
        if False:
            return 10
        'Registering a core key/value will allow it to be shown by the\n        config_option_show action.'
        key = 'ckan.datasets_per_page'
        value = 5
        params = {key: value}
        helpers.call_action('config_option_update', **params)
        show_value = helpers.call_action('config_option_show', key='ckan.datasets_per_page')
        assert show_value == value

    def test_update_registered_external_value_in_list(self):
        if False:
            return 10
        'Registering an external key/value will allow it to be included in\n        the list returned by config_option_list action.'
        key = 'ckanext.example_iconfigurer.test_conf'
        value = 'Test value'
        params = {key: value}
        helpers.call_action('config_option_update', **params)
        option_list = helpers.call_action('config_option_list')
        assert key in option_list

    def test_update_registered_external_value_in_show(self):
        if False:
            while True:
                i = 10
        'Registering an external key/value will allow it to be shown by the\n        config_option_show action.'
        key = 'ckanext.example_iconfigurer.test_conf'
        value = 'Test value'
        params = {key: value}
        helpers.call_action('config_option_update', **params)
        show_value = helpers.call_action('config_option_show', key='ckanext.example_iconfigurer.test_conf')
        assert show_value == value
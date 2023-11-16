from __future__ import annotations
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from unittest import mock
import pytest
from airflow.hooks.base import BaseHook
from airflow.listeners.listener import get_listener_manager
from airflow.plugins_manager import AirflowPlugin
from airflow.utils.module_loading import qualname
from airflow.www import app as application
from setup import AIRFLOW_SOURCES_ROOT
from tests.test_utils.config import conf_vars
from tests.test_utils.mock_plugins import mock_plugin_manager
pytestmark = pytest.mark.db_test
importlib_metadata_string = 'importlib_metadata'
try:
    import importlib_metadata
except ImportError:
    try:
        import importlib.metadata
        importlib_metadata = 'importlib.metadata'
    except ImportError:
        raise Exception('Either importlib_metadata must be installed or importlib.metadata must be available in system libraries (Python 3.9+). We seem to have neither.')
ON_LOAD_EXCEPTION_PLUGIN = '\nfrom airflow.plugins_manager import AirflowPlugin\n\nclass AirflowTestOnLoadExceptionPlugin(AirflowPlugin):\n    name = \'preload\'\n\n    def on_load(self, *args, **kwargs):\n        raise Exception("oops")\n'

@pytest.fixture(autouse=True, scope='module')
def clean_plugins():
    if False:
        i = 10
        return i + 15
    get_listener_manager().clear()
    yield
    get_listener_manager().clear()

@pytest.mark.db_test
class TestPluginsRBAC:

    @pytest.fixture(autouse=True)
    def _set_attrs(self, app):
        if False:
            while True:
                i = 10
        self.app = app
        self.appbuilder = app.appbuilder

    def test_flaskappbuilder_views(self):
        if False:
            return 10
        from tests.plugins.test_plugin import v_appbuilder_package
        appbuilder_class_name = str(v_appbuilder_package['view'].__class__.__name__)
        plugin_views = [view for view in self.appbuilder.baseviews if view.blueprint.name == appbuilder_class_name]
        assert len(plugin_views) == 1
        links = [menu_item for menu_item in self.appbuilder.menu.menu if menu_item.name == v_appbuilder_package['category']]
        assert len(links) == 1
        link = links[0]
        assert link.name == v_appbuilder_package['category']
        assert link.childs[0].name == v_appbuilder_package['name']

    def test_flaskappbuilder_menu_links(self):
        if False:
            while True:
                i = 10
        from tests.plugins.test_plugin import appbuilder_mitem, appbuilder_mitem_toplevel
        categories = [menu_item for menu_item in self.appbuilder.menu.menu if menu_item.name == appbuilder_mitem['category']]
        assert len(categories) == 1
        category = categories[0]
        assert category.name == appbuilder_mitem['category']
        assert category.childs[0].name == appbuilder_mitem['name']
        assert category.childs[0].href == appbuilder_mitem['href']
        top_levels = [menu_item for menu_item in self.appbuilder.menu.menu if menu_item.name == appbuilder_mitem_toplevel['name']]
        assert len(top_levels) == 1
        link = top_levels[0]
        assert link.href == appbuilder_mitem_toplevel['href']
        assert link.label == appbuilder_mitem_toplevel['label']

    def test_app_blueprints(self):
        if False:
            print('Hello World!')
        from tests.plugins.test_plugin import bp
        assert 'test_plugin' in self.app.blueprints
        assert self.app.blueprints['test_plugin'].name == bp.name

    def test_app_static_folder(self):
        if False:
            return 10
        assert AIRFLOW_SOURCES_ROOT / 'airflow' / 'www' / 'static' == Path(self.app.static_folder).resolve()

@pytest.mark.db_test
def test_flaskappbuilder_nomenu_views():
    if False:
        print('Hello World!')
    from tests.plugins.test_plugin import v_nomenu_appbuilder_package

    class AirflowNoMenuViewsPlugin(AirflowPlugin):
        appbuilder_views = [v_nomenu_appbuilder_package]
    appbuilder_class_name = str(v_nomenu_appbuilder_package['view'].__class__.__name__)
    with mock_plugin_manager(plugins=[AirflowNoMenuViewsPlugin()]):
        appbuilder = application.create_app(testing=True).appbuilder
        plugin_views = [view for view in appbuilder.baseviews if view.blueprint.name == appbuilder_class_name]
        assert len(plugin_views) == 1

class TestPluginsManager:

    @pytest.fixture(autouse=True, scope='function')
    def clean_plugins(self):
        if False:
            i = 10
            return i + 15
        from airflow import plugins_manager
        plugins_manager.loaded_plugins = set()
        plugins_manager.plugins = []

    def test_no_log_when_no_plugins(self, caplog):
        if False:
            print('Hello World!')
        with mock_plugin_manager(plugins=[]):
            from airflow import plugins_manager
            plugins_manager.ensure_plugins_loaded()
        assert caplog.record_tuples == []

    def test_should_load_plugins_from_property(self, caplog):
        if False:
            i = 10
            return i + 15

        class AirflowTestPropertyPlugin(AirflowPlugin):
            name = 'test_property_plugin'

            @property
            def hooks(self):
                if False:
                    print('Hello World!')

                class TestPropertyHook(BaseHook):
                    pass
                return [TestPropertyHook]
        with mock_plugin_manager(plugins=[AirflowTestPropertyPlugin()]):
            from airflow import plugins_manager
            caplog.set_level(logging.DEBUG, 'airflow.plugins_manager')
            plugins_manager.ensure_plugins_loaded()
            assert 'AirflowTestPropertyPlugin' in str(plugins_manager.plugins)
            assert 'TestPropertyHook' in str(plugins_manager.registered_hooks)
        assert caplog.records[-1].levelname == 'DEBUG'
        assert caplog.records[-1].msg == 'Loading %d plugin(s) took %.2f seconds'

    def test_loads_filesystem_plugins(self, caplog):
        if False:
            return 10
        from airflow import plugins_manager
        with mock.patch('airflow.plugins_manager.plugins', []):
            plugins_manager.load_plugins_from_plugin_directory()
            assert 6 == len(plugins_manager.plugins)
            for plugin in plugins_manager.plugins:
                if 'AirflowTestOnLoadPlugin' in str(plugin):
                    assert 'postload' == plugin.name
                    break
            else:
                pytest.fail("Wasn't able to find a registered `AirflowTestOnLoadPlugin`")
            assert caplog.record_tuples == []

    def test_loads_filesystem_plugins_exception(self, caplog, tmp_path):
        if False:
            while True:
                i = 10
        from airflow import plugins_manager
        with mock.patch('airflow.plugins_manager.plugins', []):
            (tmp_path / 'testplugin.py').write_text(ON_LOAD_EXCEPTION_PLUGIN)
            with conf_vars({('core', 'plugins_folder'): os.fspath(tmp_path)}):
                plugins_manager.load_plugins_from_plugin_directory()
            assert plugins_manager.plugins == []
            received_logs = caplog.text
            assert 'Failed to import plugin' in received_logs
            assert 'testplugin.py' in received_logs

    def test_should_warning_about_incompatible_plugins(self, caplog):
        if False:
            i = 10
            return i + 15

        class AirflowAdminViewsPlugin(AirflowPlugin):
            name = 'test_admin_views_plugin'
            admin_views = [mock.MagicMock()]

        class AirflowAdminMenuLinksPlugin(AirflowPlugin):
            name = 'test_menu_links_plugin'
            menu_links = [mock.MagicMock()]
        with mock_plugin_manager(plugins=[AirflowAdminViewsPlugin(), AirflowAdminMenuLinksPlugin()]), caplog.at_level(logging.WARNING, logger='airflow.plugins_manager'):
            from airflow import plugins_manager
            plugins_manager.initialize_web_ui_plugins()
        assert caplog.record_tuples == [('airflow.plugins_manager', logging.WARNING, "Plugin 'test_admin_views_plugin' may not be compatible with the current Airflow version. Please contact the author of the plugin."), ('airflow.plugins_manager', logging.WARNING, "Plugin 'test_menu_links_plugin' may not be compatible with the current Airflow version. Please contact the author of the plugin.")]

    def test_should_not_warning_about_fab_plugins(self, caplog):
        if False:
            print('Hello World!')

        class AirflowAdminViewsPlugin(AirflowPlugin):
            name = 'test_admin_views_plugin'
            appbuilder_views = [mock.MagicMock()]

        class AirflowAdminMenuLinksPlugin(AirflowPlugin):
            name = 'test_menu_links_plugin'
            appbuilder_menu_items = [mock.MagicMock()]
        with mock_plugin_manager(plugins=[AirflowAdminViewsPlugin(), AirflowAdminMenuLinksPlugin()]), caplog.at_level(logging.WARNING, logger='airflow.plugins_manager'):
            from airflow import plugins_manager
            plugins_manager.initialize_web_ui_plugins()
        assert caplog.record_tuples == []

    def test_should_not_warning_about_fab_and_flask_admin_plugins(self, caplog):
        if False:
            return 10

        class AirflowAdminViewsPlugin(AirflowPlugin):
            name = 'test_admin_views_plugin'
            admin_views = [mock.MagicMock()]
            appbuilder_views = [mock.MagicMock()]

        class AirflowAdminMenuLinksPlugin(AirflowPlugin):
            name = 'test_menu_links_plugin'
            menu_links = [mock.MagicMock()]
            appbuilder_menu_items = [mock.MagicMock()]
        with mock_plugin_manager(plugins=[AirflowAdminViewsPlugin(), AirflowAdminMenuLinksPlugin()]), caplog.at_level(logging.WARNING, logger='airflow.plugins_manager'):
            from airflow import plugins_manager
            plugins_manager.initialize_web_ui_plugins()
        assert caplog.record_tuples == []

    def test_entrypoint_plugin_errors_dont_raise_exceptions(self, caplog):
        if False:
            while True:
                i = 10
        '\n        Test that Airflow does not raise an error if there is any Exception because of a plugin.\n        '
        from airflow.plugins_manager import import_errors, load_entrypoint_plugins
        mock_dist = mock.Mock()
        mock_dist.metadata = {'Name': 'test-dist'}
        mock_entrypoint = mock.Mock()
        mock_entrypoint.name = 'test-entrypoint'
        mock_entrypoint.group = 'airflow.plugins'
        mock_entrypoint.module = 'test.plugins.test_plugins_manager'
        mock_entrypoint.load.side_effect = ImportError('my_fake_module not found')
        mock_dist.entry_points = [mock_entrypoint]
        with mock.patch(f'{importlib_metadata_string}.distributions', return_value=[mock_dist]), caplog.at_level(logging.ERROR, logger='airflow.plugins_manager'):
            load_entrypoint_plugins()
            received_logs = caplog.text
            assert 'Traceback (most recent call last):' in received_logs
            assert 'my_fake_module not found' in received_logs
            assert 'Failed to import plugin test-entrypoint' in received_logs
            assert ('test.plugins.test_plugins_manager', 'my_fake_module not found') in import_errors.items()

    def test_registering_plugin_macros(self, request):
        if False:
            i = 10
            return i + 15
        '\n        Tests whether macros that originate from plugins are being registered correctly.\n        '
        from airflow import macros
        from airflow.plugins_manager import integrate_macros_plugins

        def cleanup_macros():
            if False:
                return 10
            'Reloads the airflow.macros module such that the symbol table is reset after the test.'
            del sys.modules['airflow.macros']
            importlib.import_module('airflow.macros')
        request.addfinalizer(cleanup_macros)

        def custom_macro():
            if False:
                i = 10
                return i + 15
            return 'foo'

        class MacroPlugin(AirflowPlugin):
            name = 'macro_plugin'
            macros = [custom_macro]
        with mock_plugin_manager(plugins=[MacroPlugin()]):
            integrate_macros_plugins()
            plugin_macros = importlib.import_module(f'airflow.macros.{MacroPlugin.name}')
            for macro in MacroPlugin.macros:
                assert hasattr(plugin_macros, macro.__name__)
            assert hasattr(macros, MacroPlugin.name)

    def test_registering_plugin_listeners(self):
        if False:
            print('Hello World!')
        from airflow import plugins_manager
        with mock.patch('airflow.plugins_manager.plugins', []):
            plugins_manager.load_plugins_from_plugin_directory()
            plugins_manager.integrate_listener_plugins(get_listener_manager())
            assert get_listener_manager().has_listeners
            listeners = get_listener_manager().pm.get_plugins()
            listener_names = [el.__name__ if inspect.ismodule(el) else qualname(el) for el in listeners]
            assert ['tests.listeners.class_listener.ClassBasedListener', 'tests.listeners.empty_listener'] == sorted(listener_names)

    def test_should_import_plugin_from_providers(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow import plugins_manager
        with mock.patch('airflow.plugins_manager.plugins', []):
            assert len(plugins_manager.plugins) == 0
            plugins_manager.load_providers_plugins()
            assert len(plugins_manager.plugins) >= 2

    def test_does_not_double_import_entrypoint_provider_plugins(self):
        if False:
            i = 10
            return i + 15
        from airflow import plugins_manager
        mock_entrypoint = mock.Mock()
        mock_entrypoint.name = 'test-entrypoint-plugin'
        mock_entrypoint.module = 'module_name_plugin'
        mock_dist = mock.Mock()
        mock_dist.metadata = {'Name': 'test-entrypoint-plugin'}
        mock_dist.version = '1.0.0'
        mock_dist.entry_points = [mock_entrypoint]
        with mock.patch('airflow.plugins_manager.plugins', []):
            assert len(plugins_manager.plugins) == 0
            plugins_manager.load_entrypoint_plugins()
            plugins_manager.load_providers_plugins()
            assert len(plugins_manager.plugins) == 2

class TestPluginsDirectorySource:

    def test_should_return_correct_path_name(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow import plugins_manager
        source = plugins_manager.PluginsDirectorySource(__file__)
        assert 'test_plugins_manager.py' == source.path
        assert '$PLUGINS_FOLDER/test_plugins_manager.py' == str(source)
        assert '<em>$PLUGINS_FOLDER/</em>test_plugins_manager.py' == source.__html__()

class TestEntryPointSource:

    def test_should_return_correct_source_details(self):
        if False:
            print('Hello World!')
        from airflow import plugins_manager
        mock_entrypoint = mock.Mock()
        mock_entrypoint.name = 'test-entrypoint-plugin'
        mock_entrypoint.module = 'module_name_plugin'
        mock_dist = mock.Mock()
        mock_dist.metadata = {'Name': 'test-entrypoint-plugin'}
        mock_dist.version = '1.0.0'
        mock_dist.entry_points = [mock_entrypoint]
        with mock.patch(f'{importlib_metadata_string}.distributions', return_value=[mock_dist]):
            plugins_manager.load_entrypoint_plugins()
        source = plugins_manager.EntryPointSource(mock_entrypoint, mock_dist)
        assert str(mock_entrypoint) == source.entrypoint
        assert 'test-entrypoint-plugin==1.0.0: ' + str(mock_entrypoint) == str(source)
        assert '<em>test-entrypoint-plugin==1.0.0:</em> ' + str(mock_entrypoint) == source.__html__()
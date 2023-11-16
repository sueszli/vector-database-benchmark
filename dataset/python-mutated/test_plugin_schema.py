from __future__ import annotations
from flask import Blueprint
from flask_appbuilder import BaseView
from airflow.api_connexion.schemas.plugin_schema import PluginCollection, plugin_collection_schema, plugin_schema
from airflow.hooks.base import BaseHook
from airflow.models.baseoperator import BaseOperatorLink
from airflow.plugins_manager import AirflowPlugin

class PluginHook(BaseHook):
    ...

def plugin_macro():
    if False:
        for i in range(10):
            print('nop')
    ...

class MockOperatorLink(BaseOperatorLink):
    name = 'mock_operator_link'

    def get_link(self, operator, *, ti_key) -> str:
        if False:
            print('Hello World!')
        return 'mock_operator_link'
bp = Blueprint('mock_blueprint', __name__, url_prefix='/mock_blueprint')

class MockView(BaseView):
    ...
appbuilder_menu_items = {'name': 'mock_plugin', 'href': 'https://example.com'}

class MockPlugin(AirflowPlugin):
    name = 'mock_plugin'
    flask_blueprints = [bp]
    appbuilder_views = [{'view': MockView()}]
    appbuilder_menu_items = [appbuilder_menu_items]
    global_operator_extra_links = [MockOperatorLink()]
    operator_extra_links = [MockOperatorLink()]
    hooks = [PluginHook]
    macros = [plugin_macro]

class TestPluginBase:

    def setup_method(self) -> None:
        if False:
            print('Hello World!')
        self.mock_plugin = MockPlugin()
        self.mock_plugin.name = 'test_plugin'
        self.mock_plugin_2 = MockPlugin()
        self.mock_plugin_2.name = 'test_plugin_2'

class TestPluginSchema(TestPluginBase):

    def test_serialize(self):
        if False:
            print('Hello World!')
        deserialized_plugin = plugin_schema.dump(self.mock_plugin)
        assert deserialized_plugin == {'appbuilder_menu_items': [appbuilder_menu_items], 'appbuilder_views': [{'view': self.mock_plugin.appbuilder_views[0]['view']}], 'executors': [], 'flask_blueprints': [str(bp)], 'global_operator_extra_links': [str(MockOperatorLink())], 'hooks': [str(PluginHook)], 'macros': [str(plugin_macro)], 'operator_extra_links': [str(MockOperatorLink())], 'source': None, 'name': 'test_plugin', 'ti_deps': [], 'listeners': [], 'timetables': []}

class TestPluginCollectionSchema(TestPluginBase):

    def test_serialize(self):
        if False:
            while True:
                i = 10
        plugins = [self.mock_plugin, self.mock_plugin_2]
        deserialized = plugin_collection_schema.dump(PluginCollection(plugins=plugins, total_entries=2))
        assert deserialized == {'plugins': [{'appbuilder_menu_items': [appbuilder_menu_items], 'appbuilder_views': [{'view': self.mock_plugin.appbuilder_views[0]['view']}], 'executors': [], 'flask_blueprints': [str(bp)], 'global_operator_extra_links': [str(MockOperatorLink())], 'hooks': [str(PluginHook)], 'macros': [str(plugin_macro)], 'operator_extra_links': [str(MockOperatorLink())], 'source': None, 'name': 'test_plugin', 'ti_deps': [], 'listeners': [], 'timetables': []}, {'appbuilder_menu_items': [appbuilder_menu_items], 'appbuilder_views': [{'view': self.mock_plugin.appbuilder_views[0]['view']}], 'executors': [], 'flask_blueprints': [str(bp)], 'global_operator_extra_links': [str(MockOperatorLink())], 'hooks': [str(PluginHook)], 'macros': [str(plugin_macro)], 'operator_extra_links': [str(MockOperatorLink())], 'source': None, 'name': 'test_plugin_2', 'ti_deps': [], 'listeners': [], 'timetables': []}], 'total_entries': 2}
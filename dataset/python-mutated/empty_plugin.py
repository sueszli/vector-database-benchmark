"""Plugins example"""
from __future__ import annotations
from flask import Blueprint
from flask_appbuilder import BaseView, expose
from airflow.auth.managers.models.resource_details import AccessView
from airflow.plugins_manager import AirflowPlugin
from airflow.www.auth import has_access_view

class EmptyPluginView(BaseView):
    """Creating a Flask-AppBuilder View"""
    default_view = 'index'

    @expose('/')
    @has_access_view(AccessView.PLUGINS)
    def index(self):
        if False:
            print('Hello World!')
        'Create default view'
        return self.render_template('empty_plugin/index.html', name='Empty Plugin')
bp = Blueprint('Empty Plugin', __name__, template_folder='templates', static_folder='static', static_url_path='/static/empty_plugin')

class EmptyPlugin(AirflowPlugin):
    """Defining the plugin class"""
    name = 'Empty Plugin'
    flask_blueprints = [bp]
    appbuilder_views = [{'name': 'Empty Plugin', 'category': 'Extra Views', 'view': EmptyPluginView()}]
from __future__ import annotations
import pytest
from airflow.www.app import create_app
from tests.test_utils.config import conf_vars
from tests.test_utils.decorators import dont_initialize_flask_app_submodules
from tests.test_utils.www import client_with_login
pytestmark = pytest.mark.db_test

@pytest.fixture()
def app_with_rate_limit_one(examples_dag_bag):
    if False:
        for i in range(10):
            print('nop')

    @dont_initialize_flask_app_submodules(skip_all_except=['init_api_connexion', 'init_appbuilder', 'init_appbuilder_links', 'init_appbuilder_views', 'init_flash_views', 'init_jinja_globals', 'init_plugins', 'init_airflow_session_interface', 'init_check_user_active'])
    def factory():
        if False:
            i = 10
            return i + 15
        with conf_vars({('webserver', 'auth_rate_limited'): 'True', ('webserver', 'auth_rate_limit'): '1 per 20 second'}):
            return create_app(testing=True)
    app = factory()
    app.config['WTF_CSRF_ENABLED'] = False
    return app

def test_rate_limit_one(app_with_rate_limit_one):
    if False:
        i = 10
        return i + 15
    client_with_login(app_with_rate_limit_one, expected_response_code=302, username='test_admin', password='test_admin')
    client_with_login(app_with_rate_limit_one, expected_response_code=429, username='test_admin', password='test_admin')
    client_with_login(app_with_rate_limit_one, expected_response_code=429, username='test_admin', password='test_admin')

def test_rate_limit_disabled(app):
    if False:
        i = 10
        return i + 15
    client_with_login(app, expected_response_code=302, username='test_admin', password='test_admin')
    client_with_login(app, expected_response_code=302, username='test_admin', password='test_admin')
    client_with_login(app, expected_response_code=302, username='test_admin', password='test_admin')
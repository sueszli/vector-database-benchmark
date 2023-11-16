from __future__ import annotations
import html
import pytest
from airflow.configuration import conf
from tests.test_utils.config import conf_vars
from tests.test_utils.www import check_content_in_response, check_content_not_in_response
pytestmark = pytest.mark.db_test

@conf_vars({('webserver', 'expose_config'): 'False'})
def test_user_cant_view_configuration(admin_client):
    if False:
        for i in range(10):
            print('nop')
    resp = admin_client.get('configuration', follow_redirects=True)
    check_content_in_response('Your Airflow administrator chose not to expose the configuration, most likely for security reasons.', resp)

@conf_vars({('webserver', 'expose_config'): 'True'})
def test_user_can_view_configuration(admin_client):
    if False:
        print('Hello World!')
    resp = admin_client.get('configuration', follow_redirects=True)
    for (section, key) in conf.sensitive_config_values:
        value = conf.get(section, key, fallback='')
        if value:
            check_content_in_response(html.escape(value), resp)

@conf_vars({('webserver', 'expose_config'): 'non-sensitive-only'})
def test_configuration_redacted(admin_client):
    if False:
        return 10
    resp = admin_client.get('configuration', follow_redirects=True)
    for (section, key) in conf.sensitive_config_values:
        value = conf.get(section, key, fallback='')
        if value and value != 'airflow' and (not value.startswith('db+postgresql')):
            check_content_not_in_response(value, resp)

@conf_vars({('webserver', 'expose_config'): 'non-sensitive-only'})
def test_configuration_redacted_in_running_configuration(admin_client):
    if False:
        print('Hello World!')
    resp = admin_client.get('configuration', follow_redirects=True)
    for (section, key) in conf.sensitive_config_values:
        value = conf.get(section, key, fallback='')
        if value and value != 'airflow':
            check_content_not_in_response("<td class='code'>" + html.escape(value) + '</td', resp)

@conf_vars({('webserver', 'expose_config'): 'non-sensitive-only'})
@conf_vars({('database', '# sql_alchemy_conn'): 'testconn'})
@conf_vars({('core', '  # secret_key'): 'core_secret'})
@conf_vars({('core', 'fernet_key'): 'secret_fernet_key'})
def test_commented_out_config(admin_client):
    if False:
        while True:
            i = 10
    resp = admin_client.get('configuration', follow_redirects=True)
    check_content_in_response('testconn', resp)
    check_content_in_response('core_secret', resp)
    check_content_not_in_response('secret_fernet_key', resp)
from __future__ import annotations
import json
import os
from datetime import datetime
from unittest import mock
import pytest
from airflow.api.auth.backend.kerberos_auth import CLIENT_AUTH
from airflow.models import DagBag
from airflow.utils.net import getfqdn
from airflow.www import app
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_dags
KRB5_KTNAME = os.environ.get('KRB5_KTNAME')

@pytest.fixture(scope='module')
def app_for_kerberos():
    if False:
        i = 10
        return i + 15
    with conf_vars({('api', 'auth_backends'): 'airflow.api.auth.backend.kerberos_auth', ('kerberos', 'keytab'): KRB5_KTNAME, ('api', 'enable_experimental_api'): 'true'}):
        yield app.create_app(testing=True)

@pytest.fixture(scope='module')
def dagbag_to_db():
    if False:
        for i in range(10):
            print('nop')
    DagBag(include_examples=True).sync_to_db()
    yield
    clear_db_dags()

@pytest.mark.integration('kerberos')
class TestApiKerberos:

    @pytest.fixture(autouse=True)
    def _set_attrs(self, app_for_kerberos, dagbag_to_db):
        if False:
            i = 10
            return i + 15
        self.app = app_for_kerberos

    def test_trigger_dag(self):
        if False:
            print('Hello World!')
        with self.app.test_client() as client:
            url_template = '/api/experimental/dags/{}/dag_runs'
            response = client.post(url_template.format('example_bash_operator'), data=json.dumps(dict(run_id='my_run' + datetime.now().isoformat())), content_type='application/json')
            assert 401 == response.status_code
            response.url = f'http://{getfqdn()}'

            class Request:
                headers = {}
            response.request = Request()
            response.content = ''
            response.raw = mock.MagicMock()
            response.connection = mock.MagicMock()
            response.connection.send = mock.MagicMock()
            CLIENT_AUTH.mutual_authentication = 3
            CLIENT_AUTH.handle_response(response)
            assert 'Authorization' in response.request.headers
            response2 = client.post(url_template.format('example_bash_operator'), data=json.dumps(dict(run_id='my_run' + datetime.now().isoformat())), content_type='application/json', headers=response.request.headers)
            assert 200 == response2.status_code

    def test_unauthorized(self):
        if False:
            print('Hello World!')
        with self.app.test_client() as client:
            url_template = '/api/experimental/dags/{}/dag_runs'
            response = client.post(url_template.format('example_bash_operator'), data=json.dumps(dict(run_id='my_run' + datetime.now().isoformat())), content_type='application/json')
            assert 401 == response.status_code
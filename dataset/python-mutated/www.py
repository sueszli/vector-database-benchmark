from __future__ import annotations
import ast
from unittest import mock
from airflow.models import Log

def client_with_login(app, expected_response_code=302, **kwargs):
    if False:
        print('Hello World!')
    patch_path = 'airflow.auth.managers.fab.security_manager.override.check_password_hash'
    with mock.patch(patch_path) as check_password_hash:
        check_password_hash.return_value = True
        client = app.test_client()
        resp = client.post('/login/', data=kwargs)
        assert resp.status_code == expected_response_code
    return client

def client_without_login(app):
    if False:
        print('Hello World!')
    app.config['AUTH_ROLE_PUBLIC'] = 'Viewer'
    client = app.test_client()
    return client

def check_content_in_response(text, resp, resp_code=200):
    if False:
        print('Hello World!')
    resp_html = resp.data.decode('utf-8')
    assert resp_code == resp.status_code
    if isinstance(text, list):
        for line in text:
            assert line in resp_html, f"Couldn't find {line!r}"
    else:
        assert text in resp_html, f"Couldn't find {text!r}"

def check_content_not_in_response(text, resp, resp_code=200):
    if False:
        return 10
    resp_html = resp.data.decode('utf-8')
    assert resp_code == resp.status_code
    if isinstance(text, list):
        for line in text:
            assert line not in resp_html
    else:
        assert text not in resp_html

def _check_last_log(session, dag_id, event, execution_date):
    if False:
        return 10
    logs = session.query(Log.dag_id, Log.task_id, Log.event, Log.execution_date, Log.owner, Log.extra).filter(Log.dag_id == dag_id, Log.event == event, Log.execution_date == execution_date).order_by(Log.dttm.desc()).limit(5).all()
    assert len(logs) >= 1
    assert logs[0].extra
    session.query(Log).delete()

def _check_last_log_masked_connection(session, dag_id, event, execution_date):
    if False:
        while True:
            i = 10
    logs = session.query(Log.dag_id, Log.task_id, Log.event, Log.execution_date, Log.owner, Log.extra).filter(Log.dag_id == dag_id, Log.event == event, Log.execution_date == execution_date).order_by(Log.dttm.desc()).limit(5).all()
    assert len(logs) >= 1
    extra = ast.literal_eval(logs[0].extra)
    assert extra == [('conn_id', 'test_conn'), ('conn_type', 'http'), ('description', 'description'), ('host', 'localhost'), ('port', '8080'), ('username', 'root'), ('password', '***'), ('extra', '{"x_secret": "***", "y_secret": "***"}')]

def _check_last_log_masked_variable(session, dag_id, event, execution_date):
    if False:
        while True:
            i = 10
    logs = session.query(Log.dag_id, Log.task_id, Log.event, Log.execution_date, Log.owner, Log.extra).filter(Log.dag_id == dag_id, Log.event == event, Log.execution_date == execution_date).order_by(Log.dttm.desc()).limit(5).all()
    assert len(logs) >= 1
    extra_dict = ast.literal_eval(logs[0].extra)
    assert extra_dict == [('key', 'x_secret'), ('val', '***')]
from __future__ import annotations
import datetime
import json
import pytest
from airflow.jobs.job import Job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.utils import timezone
from airflow.utils.session import create_session
from airflow.www import app as application
from tests.test_utils.asserts import assert_queries_count
from tests.test_utils.config import conf_vars
from tests.test_utils.www import check_content_in_response, check_content_not_in_response
pytestmark = pytest.mark.db_test

def test_index_redirect(admin_client):
    if False:
        return 10
    resp = admin_client.get('/')
    assert resp.status_code == 302
    assert '/home' in resp.headers.get('Location')
    resp = admin_client.get('/', follow_redirects=True)
    check_content_in_response('DAGs', resp)

def test_homepage_query_count(admin_client):
    if False:
        for i in range(10):
            print('nop')
    with assert_queries_count(17):
        resp = admin_client.get('/home')
    check_content_in_response('DAGs', resp)

def test_doc_urls(admin_client, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    from airflow.www.views import AirflowBaseView
    monkeypatch.setitem(AirflowBaseView.extra_args, 'get_docs_url', lambda _: '!!DOCS_URL!!')
    resp = admin_client.get('/', follow_redirects=True)
    check_content_in_response('!!DOCS_URL!!', resp)
    check_content_in_response('/api/v1/ui', resp)

@pytest.fixture()
def heartbeat_healthy():
    if False:
        i = 10
        return i + 15
    last_heartbeat = timezone.utcnow()
    job = Job(state='running', latest_heartbeat=last_heartbeat)
    SchedulerJobRunner(job=job)
    with create_session() as session:
        session.add(job)
    yield ('healthy', last_heartbeat.isoformat())
    with create_session() as session:
        session.query(Job).filter(Job.job_type == 'SchedulerJob', Job.state == 'running', Job.latest_heartbeat == last_heartbeat).delete()

@pytest.fixture()
def heartbeat_too_slow():
    if False:
        while True:
            i = 10
    last_heartbeat = timezone.utcnow() - datetime.timedelta(minutes=1)
    job = Job(state='running', latest_heartbeat=last_heartbeat)
    SchedulerJobRunner(job=job)
    with create_session() as session:
        session.query(Job).filter(Job.job_type == 'SchedulerJob').update({'latest_heartbeat': last_heartbeat - datetime.timedelta(seconds=1)})
        session.add(job)
    yield ('unhealthy', last_heartbeat.isoformat())
    with create_session() as session:
        session.query(Job).filter(Job.job_type == 'SchedulerJob', Job.state == 'running', Job.latest_heartbeat == last_heartbeat).delete()

@pytest.fixture()
def heartbeat_not_running():
    if False:
        while True:
            i = 10
    with create_session() as session:
        session.query(Job).filter(Job.job_type == 'SchedulerJob', Job.state == 'running').delete()
    yield ('unhealthy', None)

@pytest.mark.parametrize('heartbeat', ['heartbeat_healthy', 'heartbeat_too_slow', 'heartbeat_not_running'])
def test_health(request, admin_client, heartbeat):
    if False:
        i = 10
        return i + 15
    (scheduler_status, last_scheduler_heartbeat) = request.getfixturevalue(heartbeat)
    resp = admin_client.get('health', follow_redirects=True)
    resp_json = json.loads(resp.data.decode('utf-8'))
    assert 'healthy' == resp_json['metadatabase']['status']
    assert scheduler_status == resp_json['scheduler']['status']
    assert last_scheduler_heartbeat == resp_json['scheduler']['latest_scheduler_heartbeat']

def test_users_list(admin_client):
    if False:
        while True:
            i = 10
    resp = admin_client.get('users/list', follow_redirects=True)
    check_content_in_response('List Users', resp)

@pytest.mark.parametrize('path, body_content', [('roles/list', 'List Roles'), ('roles/show/1', 'Show Role')])
def test_roles_read(admin_client, path, body_content):
    if False:
        i = 10
        return i + 15
    resp = admin_client.get(path, follow_redirects=True)
    check_content_in_response(body_content, resp)

def test_roles_read_unauthorized(viewer_client):
    if False:
        while True:
            i = 10
    resp = viewer_client.get('roles/list', follow_redirects=True)
    check_content_in_response('Access is Denied', resp)

@pytest.fixture(scope='module')
def delete_role_if_exists(app):
    if False:
        return 10

    def func(role_name):
        if False:
            return 10
        if app.appbuilder.sm.find_role(role_name):
            app.appbuilder.sm.delete_role(role_name)
    return func

@pytest.fixture()
def non_exist_role_name(delete_role_if_exists):
    if False:
        return 10
    role_name = 'test_roles_create_role'
    delete_role_if_exists(role_name)
    yield role_name
    delete_role_if_exists(role_name)

@pytest.fixture()
def exist_role_name(app, delete_role_if_exists):
    if False:
        while True:
            i = 10
    role_name = 'test_roles_create_role_new'
    app.appbuilder.sm.add_role(role_name)
    yield role_name
    delete_role_if_exists(role_name)

@pytest.fixture()
def exist_role(app, exist_role_name):
    if False:
        print('Hello World!')
    return app.appbuilder.sm.find_role(exist_role_name)

def test_roles_create(app, admin_client, non_exist_role_name):
    if False:
        return 10
    admin_client.post('roles/add', data={'name': non_exist_role_name}, follow_redirects=True)
    assert app.appbuilder.sm.find_role(non_exist_role_name) is not None

def test_roles_create_unauthorized(app, viewer_client, non_exist_role_name):
    if False:
        for i in range(10):
            print('nop')
    resp = viewer_client.post('roles/add', data={'name': non_exist_role_name}, follow_redirects=True)
    check_content_in_response('Access is Denied', resp)
    assert app.appbuilder.sm.find_role(non_exist_role_name) is None

def test_roles_edit(app, admin_client, non_exist_role_name, exist_role):
    if False:
        while True:
            i = 10
    admin_client.post(f'roles/edit/{exist_role.id}', data={'name': non_exist_role_name}, follow_redirects=True)
    updated_role = app.appbuilder.sm.find_role(non_exist_role_name)
    assert exist_role.id == updated_role.id

def test_roles_edit_unauthorized(app, viewer_client, non_exist_role_name, exist_role_name, exist_role):
    if False:
        for i in range(10):
            print('nop')
    resp = viewer_client.post(f'roles/edit/{exist_role.id}', data={'name': non_exist_role_name}, follow_redirects=True)
    check_content_in_response('Access is Denied', resp)
    assert app.appbuilder.sm.find_role(exist_role_name)
    assert app.appbuilder.sm.find_role(non_exist_role_name) is None

def test_roles_delete(app, admin_client, exist_role_name, exist_role):
    if False:
        print('Hello World!')
    admin_client.post(f'roles/delete/{exist_role.id}', follow_redirects=True)
    assert app.appbuilder.sm.find_role(exist_role_name) is None

def test_roles_delete_unauthorized(app, viewer_client, exist_role, exist_role_name):
    if False:
        while True:
            i = 10
    resp = viewer_client.post(f'roles/delete/{exist_role.id}', follow_redirects=True)
    check_content_in_response('Access is Denied', resp)
    assert app.appbuilder.sm.find_role(exist_role_name)

@pytest.mark.parametrize('url, client, content', [('userstatschartview/chart/', 'admin_client', 'User Statistics'), ('userstatschartview/chart/', 'viewer_client', 'Access is Denied'), ('actions/list', 'admin_client', 'List Actions'), ('actions/list', 'viewer_client', 'Access is Denied'), ('resources/list/', 'admin_client', 'List Resources'), ('resources/list/', 'viewer_client', 'Access is Denied'), ('permissions/list/', 'admin_client', 'List Permissions'), ('permissions/list/', 'viewer_client', 'Access is Denied'), ('resetpassword/form?pk=1', 'admin_client', 'Reset Password Form'), ('resetpassword/form?pk=1', 'viewer_client', 'Access is Denied'), ('users/list', 'admin_client', 'List Users'), ('users/list', 'viewer_client', 'Access is Denied')], ids=['userstatschertview-admin', 'userstatschertview-viewer', 'actions-admin', 'actions-viewer', 'resources-admin', 'resources-viewer', 'permissions-admin', 'permissions-viewer', 'resetpassword-admin', 'resetpassword-viewer', 'users-admin', 'users-viewer'])
def test_views_get(request, url, client, content):
    if False:
        for i in range(10):
            print('nop')
    resp = request.getfixturevalue(client).get(url, follow_redirects=True)
    check_content_in_response(content, resp)

def _check_task_stats_json(resp):
    if False:
        return 10
    return set(next(iter(resp.json.items()))[1][0]) == {'state', 'count'}

@pytest.mark.parametrize('url, check_response', [('blocked', None), ('dag_stats', None), ('task_stats', _check_task_stats_json)])
def test_views_post(admin_client, url, check_response):
    if False:
        i = 10
        return i + 15
    resp = admin_client.post(url, follow_redirects=True)
    assert resp.status_code == 200
    if check_response:
        assert check_response(resp)

@pytest.mark.parametrize('url, client, content, username', [('resetmypassword/form', 'viewer_client', 'Password Changed', 'test_viewer'), ('resetpassword/form?pk={}', 'admin_client', 'Password Changed', 'test_admin'), ('resetpassword/form?pk={}', 'viewer_client', 'Access is Denied', 'test_viewer')], ids=['my-viewer', 'pk-admin', 'pk-viewer'])
def test_resetmypasswordview_edit(app, request, url, client, content, username):
    if False:
        for i in range(10):
            print('nop')
    user = app.appbuilder.sm.find_user(username)
    resp = request.getfixturevalue(client).post(url.format(user.id), data={'password': 'blah', 'conf_password': 'blah'}, follow_redirects=True)
    check_content_in_response(content, resp)

def test_resetmypasswordview_read(viewer_client):
    if False:
        return 10
    resp = viewer_client.get('resetmypassword/form', follow_redirects=True)
    check_content_in_response('Reset Password Form', resp)

def test_get_myuserinfo(admin_client):
    if False:
        return 10
    resp = admin_client.get('users/userinfo/', follow_redirects=True)
    check_content_in_response('Your user information', resp)

def test_edit_myuserinfo(admin_client):
    if False:
        for i in range(10):
            print('nop')
    resp = admin_client.post('userinfoeditview/form', data={'first_name': 'new_first_name', 'last_name': 'new_last_name'}, follow_redirects=True)
    check_content_in_response('User information changed', resp)

@pytest.mark.parametrize('url', ['users/add', 'users/edit/1', 'users/delete/1'], ids=['add-user', 'edit-user', 'delete-user'])
def test_views_post_access_denied(viewer_client, url):
    if False:
        return 10
    resp = viewer_client.get(url, follow_redirects=True)
    check_content_in_response('Access is Denied', resp)

@pytest.fixture()
def non_exist_username(app):
    if False:
        while True:
            i = 10
    username = 'fake_username'
    user = app.appbuilder.sm.find_user(username)
    if user is not None:
        app.appbuilder.sm.del_register_user(user)
    yield username
    user = app.appbuilder.sm.find_user(username)
    if user is not None:
        app.appbuilder.sm.del_register_user(user)

def test_create_user(app, admin_client, non_exist_username):
    if False:
        i = 10
        return i + 15
    resp = admin_client.post('users/add', data={'first_name': 'fake_first_name', 'last_name': 'fake_last_name', 'username': non_exist_username, 'email': 'fake_email@email.com', 'roles': [1], 'password': 'test', 'conf_password': 'test'}, follow_redirects=True)
    check_content_in_response('Added Row', resp)
    assert app.appbuilder.sm.find_user(non_exist_username)

@pytest.fixture()
def exist_username(app, exist_role):
    if False:
        print('Hello World!')
    username = 'test_edit_user_user'
    app.appbuilder.sm.add_user(username, 'first_name', 'last_name', 'email@email.com', exist_role, password='password')
    yield username
    if app.appbuilder.sm.find_user(username):
        app.appbuilder.sm.del_register_user(username)

def test_edit_user(app, admin_client, exist_username):
    if False:
        print('Hello World!')
    user = app.appbuilder.sm.find_user(exist_username)
    resp = admin_client.post(f'users/edit/{user.id}', data={'first_name': 'new_first_name'}, follow_redirects=True)
    check_content_in_response('new_first_name', resp)

def test_delete_user(app, admin_client, exist_username):
    if False:
        for i in range(10):
            print('nop')
    user = app.appbuilder.sm.find_user(exist_username)
    resp = admin_client.post(f'users/delete/{user.id}', follow_redirects=True)
    check_content_in_response('Deleted Row', resp)

@conf_vars({('webserver', 'show_recent_stats_for_completed_runs'): 'False'})
def test_task_stats_only_noncompleted(admin_client):
    if False:
        return 10
    resp = admin_client.post('task_stats', follow_redirects=True)
    assert resp.status_code == 200

@conf_vars({('webserver', 'instance_name'): 'Site Title Test'})
def test_page_instance_name(admin_client):
    if False:
        return 10
    resp = admin_client.get('home', follow_redirects=True)
    check_content_in_response('Site Title Test', resp)

def test_page_instance_name_xss_prevention(admin_client):
    if False:
        i = 10
        return i + 15
    xss_string = "<script>alert('Give me your credit card number')</script>"
    with conf_vars({('webserver', 'instance_name'): xss_string}):
        resp = admin_client.get('home', follow_redirects=True)
        escaped_xss_string = '&lt;script&gt;alert(&#39;Give me your credit card number&#39;)&lt;/script&gt;'
        check_content_in_response(escaped_xss_string, resp)
        check_content_not_in_response(xss_string, resp)
instance_name_with_markup_conf = {('webserver', 'instance_name'): '<b>Bold Site Title Test</b>', ('webserver', 'instance_name_has_markup'): 'True'}

@conf_vars(instance_name_with_markup_conf)
def test_page_instance_name_with_markup(admin_client):
    if False:
        return 10
    resp = admin_client.get('home', follow_redirects=True)
    check_content_in_response('<b>Bold Site Title Test</b>', resp)
    check_content_not_in_response('&lt;b&gt;Bold Site Title Test&lt;/b&gt;', resp)

@conf_vars(instance_name_with_markup_conf)
def test_page_instance_name_with_markup_title():
    if False:
        i = 10
        return i + 15
    appbuilder = application.create_app(testing=True).appbuilder
    assert appbuilder.app_name == 'Bold Site Title Test'
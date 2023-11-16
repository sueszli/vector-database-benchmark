from __future__ import annotations
import flask
import markupsafe
import pytest
from airflow.models import Pool
from airflow.utils.session import create_session
from tests.test_utils.www import check_content_in_response, check_content_not_in_response
pytestmark = pytest.mark.db_test
POOL = {'pool': 'test-pool', 'slots': 777, 'description': 'test-pool-description', 'include_deferred': False}

@pytest.fixture(autouse=True)
def clear_pools():
    if False:
        print('Hello World!')
    with create_session() as session:
        session.query(Pool).delete()

@pytest.fixture()
def pool_factory(session):
    if False:
        return 10

    def factory(**values):
        if False:
            while True:
                i = 10
        pool = Pool(**{**POOL, **values})
        session.add(pool)
        session.commit()
        return pool
    return factory

def test_create_pool_with_same_name(admin_client):
    if False:
        i = 10
        return i + 15
    resp = admin_client.post('/pool/add', data=POOL, follow_redirects=True)
    check_content_in_response('Added Row', resp)
    resp = admin_client.post('/pool/add', data=POOL, follow_redirects=True)
    check_content_in_response('Already exists.', resp)

def test_create_pool_with_empty_name(admin_client):
    if False:
        i = 10
        return i + 15
    resp = admin_client.post('/pool/add', data={**POOL, 'pool': ''}, follow_redirects=True)
    check_content_in_response('This field is required.', resp)

def test_odd_name(admin_client, pool_factory):
    if False:
        i = 10
        return i + 15
    pool_factory(pool='test-pool<script></script>')
    resp = admin_client.get('/pool/list/')
    check_content_in_response('test-pool&lt;script&gt;', resp)
    check_content_not_in_response('test-pool<script>', resp)

def test_list(app, admin_client, pool_factory):
    if False:
        for i in range(10):
            print('nop')
    pool_factory(pool='test-pool')
    resp = admin_client.get('/pool/list/')
    with app.test_request_context():
        description_tag = markupsafe.Markup('<td>{description}</td>').format(description='test-pool-description')
        url = flask.url_for('TaskInstanceModelView.list', _flt_3_pool='test-pool', _flt_3_state='running')
        used_tag = markupsafe.Markup("<a href='{url}'>{slots}</a>").format(url=url, slots=0)
        url = flask.url_for('TaskInstanceModelView.list', _flt_3_pool='test-pool', _flt_3_state='queued')
        queued_tag = markupsafe.Markup("<a href='{url}'>{slots}</a>").format(url=url, slots=0)
        url = flask.url_for('TaskInstanceModelView.list', _flt_3_pool='test-pool', _flt_3_state='scheduled')
        scheduled_tag = markupsafe.Markup("<a href='{url}'>{slots}</a>").format(url=url, slots=0)
        url = flask.url_for('TaskInstanceModelView.list', _flt_3_pool='test-pool', _flt_3_state='deferred')
        deferred_tag = markupsafe.Markup("<a href='{url}'>{slots}</a>").format(url=url, slots=0)
    check_content_in_response(description_tag, resp)
    check_content_in_response(used_tag, resp)
    check_content_in_response(queued_tag, resp)
    check_content_in_response(scheduled_tag, resp)
    check_content_in_response(deferred_tag, resp)

def test_pool_muldelete(session, admin_client, pool_factory):
    if False:
        print('Hello World!')
    pool = pool_factory()
    resp = admin_client.post('/pool/action_post', data={'action': 'muldelete', 'rowid': [pool.id]}, follow_redirects=True)
    assert resp.status_code == 200
    assert session.query(Pool).filter(Pool.id == pool.id).count() == 0

def test_pool_muldelete_default(session, admin_client, pool_factory):
    if False:
        while True:
            i = 10
    pool = pool_factory(pool='default_pool')
    resp = admin_client.post('/pool/action_post', data={'action': 'muldelete', 'rowid': [pool.id]}, follow_redirects=True)
    check_content_in_response('default_pool cannot be deleted', resp)
    assert session.query(Pool).filter(Pool.id == pool.id).count() == 1

def test_pool_muldelete_access_denied(session, viewer_client, pool_factory):
    if False:
        for i in range(10):
            print('nop')
    pool = pool_factory()
    resp = viewer_client.post('/pool/action_post', data={'action': 'muldelete', 'rowid': [pool.id]}, follow_redirects=True)
    check_content_in_response('Access is Denied', resp)
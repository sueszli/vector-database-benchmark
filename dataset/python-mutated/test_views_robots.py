from __future__ import annotations
import pytest
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test

def test_robots(viewer_client):
    if False:
        for i in range(10):
            print('nop')
    resp = viewer_client.get('/robots.txt', follow_redirects=True)
    assert resp.data.decode('utf-8') == 'User-agent: *\nDisallow: /\n'

def test_deployment_warning_config(admin_client):
    if False:
        return 10
    warn_text = 'webserver.warn_deployment_exposure'
    admin_client.get('/robots.txt', follow_redirects=True)
    resp = admin_client.get('', follow_redirects=True)
    assert warn_text in resp.data.decode('utf-8')
    with conf_vars({('webserver', 'warn_deployment_exposure'): 'False'}):
        admin_client.get('/robots.txt', follow_redirects=True)
        resp = admin_client.get('/robots.txt', follow_redirects=True)
        assert warn_text not in resp.data.decode('utf-8')
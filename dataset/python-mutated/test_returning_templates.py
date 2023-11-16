from docs.examples.templating.returning_templates_jinja import app as jinja_app
from docs.examples.templating.returning_templates_jinja import app as minijinja_app
from docs.examples.templating.returning_templates_mako import app as mako_app
from litestar.testing import TestClient

def test_returning_templates_jinja():
    if False:
        i = 10
        return i + 15
    with TestClient(jinja_app) as client:
        response = client.get('/', params={'name': 'Jinja'})
        assert response.text == 'Hello <strong>Jinja</strong>'

def test_returning_templates_mako():
    if False:
        while True:
            i = 10
    with TestClient(mako_app) as client:
        response = client.get('/', params={'name': 'Mako'})
        assert response.text.strip() == 'Hello <strong>Mako</strong>'

def test_returning_templates_minijinja():
    if False:
        for i in range(10):
            print('nop')
    with TestClient(minijinja_app) as client:
        response = client.get('/', params={'name': 'Minijinja'})
        assert response.text == 'Hello <strong>Minijinja</strong>'
import pytest
import webtest
import failure

@pytest.fixture
def app(testbed):
    if False:
        i = 10
        return i + 15
    return webtest.TestApp(failure.app)

def test_get(app):
    if False:
        i = 10
        return i + 15
    app.get('/')

def test_read(app):
    if False:
        return 10
    app.get('/read')

def test_delete(app):
    if False:
        while True:
            i = 10
    app.get('/delete')
import pytest
import webtest
import backend

@pytest.fixture
def app():
    if False:
        i = 10
        return i + 15
    return webtest.TestApp(backend.app)

def test_get_module_info(app):
    if False:
        while True:
            i = 10
    result = app.get('/')
    assert result.status_code == 200
    assert 'hello world' in result.body
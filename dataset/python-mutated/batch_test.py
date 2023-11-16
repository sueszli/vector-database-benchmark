import pytest
import webtest
import batch

@pytest.fixture
def app(testbed):
    if False:
        for i in range(10):
            print('nop')
    return webtest.TestApp(batch.app)

def test_get(app):
    if False:
        print('Hello World!')
    response = app.get('/')
    assert 'Bill Holiday' in response.body
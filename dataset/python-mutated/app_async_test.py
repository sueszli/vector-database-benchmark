import pytest
import webtest
import app_async

@pytest.fixture
def app(testbed):
    if False:
        print('Hello World!')
    return webtest.TestApp(app_async.app)

def test_main(app, testbed, login):
    if False:
        print('Hello World!')
    app_async.Account(id='123', view_counter=4).put()
    login(id='123')
    response = app.get('/')
    assert response.status_int == 200
    account = app_async.Account.get_by_id('123')
    assert account.view_counter == 5
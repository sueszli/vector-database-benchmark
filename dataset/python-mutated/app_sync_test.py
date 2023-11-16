import pytest
import webtest
import app_sync

@pytest.fixture
def app(testbed):
    if False:
        i = 10
        return i + 15
    return webtest.TestApp(app_sync.app)

def test_main(app, testbed, login):
    if False:
        return 10
    app_sync.Account(id='123', view_counter=4).put()
    login(id='123')
    response = app.get('/')
    assert response.status_int == 200
    account = app_sync.Account.get_by_id('123')
    assert account.view_counter == 5
import pytest
import webtest
import app_toplevel

@pytest.fixture
def app(testbed):
    if False:
        for i in range(10):
            print('nop')
    return webtest.TestApp(app_toplevel.app)

def test_main(app, testbed, login):
    if False:
        while True:
            i = 10
    app_toplevel.Account(id='123', view_counter=4).put()
    login(id='123')
    response = app.get('/')
    assert response.status_int == 200
    account = app_toplevel.Account.get_by_id('123')
    assert account.view_counter == 5
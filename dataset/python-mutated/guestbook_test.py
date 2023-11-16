import pytest
import webtest
import guestbook

@pytest.fixture
def app(testbed):
    if False:
        print('Hello World!')
    return webtest.TestApp(guestbook.app)

def test_get_guestbook_sync(app, testbed, login):
    if False:
        for i in range(10):
            print('nop')
    guestbook.Account(id='123').put()
    login(id='123')
    for i in range(11):
        guestbook.Guestbook(content='Content {}'.format(i)).put()
    response = app.get('/guestbook')
    assert response.status_int == 200
    assert 'Content 1' in response.body

def test_get_guestbook_async(app, testbed, login):
    if False:
        for i in range(10):
            print('nop')
    guestbook.Account(id='123').put()
    login(id='123')
    for i in range(11):
        guestbook.Guestbook(content='Content {}'.format(i)).put()
    response = app.get('/guestbook?async=1')
    assert response.status_int == 200
    assert 'Content 1' in response.body

def test_get_messages_sync(app, testbed):
    if False:
        while True:
            i = 10
    for i in range(21):
        account_key = guestbook.Account(nickname='Nick {}'.format(i)).put()
        guestbook.Message(author=account_key, text='Text {}'.format(i)).put()
    response = app.get('/messages')
    assert response.status_int == 200
    assert 'Nick 1 wrote:' in response.body
    assert '<p>Text 1' in response.body

def test_get_messages_async(app, testbed):
    if False:
        i = 10
        return i + 15
    for i in range(21):
        account_key = guestbook.Account(nickname='Nick {}'.format(i)).put()
        guestbook.Message(author=account_key, text='Text {}'.format(i)).put()
    response = app.get('/messages?async=1')
    assert response.status_int == 200
    assert 'Nick 1 wrote:' in response.body
    assert '\nText 1' in response.body
import mock
import pytest
import webtest
import xmpp

@pytest.fixture
def app(testbed):
    if False:
        i = 10
        return i + 15
    return webtest.TestApp(xmpp.app)

@mock.patch('xmpp.xmpp')
def test_chat(xmpp_mock, app):
    if False:
        for i in range(10):
            print('nop')
    app.post('/_ah/xmpp/message/chat/', {'from': 'sender@example.com', 'to': 'recipient@example.com', 'body': 'hello'})

@mock.patch('xmpp.xmpp')
def test_subscribe(xmpp_mock, app):
    if False:
        i = 10
        return i + 15
    app.post('/_ah/xmpp/subscribe')

@mock.patch('xmpp.xmpp')
def test_check_presence(xmpp_mock, app):
    if False:
        while True:
            i = 10
    app.post('/_ah/xmpp/presence/available', {'from': 'sender@example.com'})

@mock.patch('xmpp.xmpp')
def test_send_presence(xmpp_mock, app):
    if False:
        i = 10
        return i + 15
    app.post('/send_presence', {'jid': 'node@domain/resource'})

@mock.patch('xmpp.xmpp')
def test_error(xmpp_mock, app):
    if False:
        print('Hello World!')
    app.post('/_ah/xmpp/error/', {'from': 'sender@example.com', 'stanza': 'hello world'})

@mock.patch('xmpp.xmpp')
def test_send_chat(xmpp_mock, app):
    if False:
        print('Hello World!')
    app.post('/send_chat')
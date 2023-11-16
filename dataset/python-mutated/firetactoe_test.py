import mock
import re
from google.appengine.api import users
from google.appengine.ext import ndb
from six.moves import http_client
import pytest
import webtest
import firetactoe

class MockResponse:

    def __init__(self, json_data, status_code):
        if False:
            while True:
                i = 10
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        if False:
            for i in range(10):
                print('nop')
        return self.json_data

@pytest.fixture
def app(testbed, monkeypatch, login):
    if False:
        return 10
    firetactoe._get_session.cache_clear()
    monkeypatch.setattr(firetactoe, '_FIREBASE_CONFIG', '../firetactoe_test.py')
    login(id='38')
    firetactoe.app.debug = True
    return webtest.TestApp(firetactoe.app)

def test_index_new_game(app, monkeypatch):
    if False:
        print('Hello World!')
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        response = app.get('/')
        assert 'g=' in response.body
        assert re.search("initGame[^\\n]+\\'[\\w+/=]+\\.[\\w+/=]+\\.[\\w+/=]+\\'", response.body)
        assert firetactoe.Game.query().count() == 1
        auth_session.assert_called_once_with(mock.ANY, method='PATCH', url='http://firebase.com/test-db-url/channels/3838.json', body='{"winner": null, "userX": "38", "moveX": true, "winningBoard": null, "board": "         ", "userO": null}', data=None)

def test_index_existing_game(app, monkeypatch):
    if False:
        return 10
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        userX = users.User('x@example.com', _user_id='123')
        firetactoe.Game(id='razem', userX=userX).put()
        response = app.get('/?g=razem')
        assert 'g=' in response.body
        assert re.search("initGame[^\\n]+\\'[\\w+/=]+\\.[\\w+/=]+\\.[\\w+/=]+\\'", response.body)
        assert firetactoe.Game.query().count() == 1
        game = ndb.Key('Game', 'razem').get()
        assert game is not None
        assert game.userO.user_id() == '38'
        auth_session.assert_called_once_with(mock.ANY, method='PATCH', url='http://firebase.com/test-db-url/channels/38razem.json', body='{"winner": null, "userX": "123", "moveX": null, "winningBoard": null, "board": null, "userO": "38"}', data=None)

def test_index_nonexisting_game(app, monkeypatch):
    if False:
        return 10
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        firetactoe.Game(id='razem', userX=users.get_current_user()).put()
        app.get('/?g=razemfrazem', status=404)
        assert not auth_session.called

def test_opened(app, monkeypatch):
    if False:
        i = 10
        return i + 15
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        firetactoe.Game(id='razem', userX=users.get_current_user()).put()
        app.post('/opened?g=razem', status=200)
        auth_session.assert_called_once_with(mock.ANY, method='PATCH', url='http://firebase.com/test-db-url/channels/38razem.json', body='{"winner": null, "userX": "38", "moveX": null, "winningBoard": null, "board": null, "userO": null}', data=None)

def test_bad_move(app, monkeypatch):
    if False:
        i = 10
        return i + 15
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        firetactoe.Game(id='razem', userX=users.get_current_user(), board=9 * ' ', moveX=True).put()
        app.post('/move?g=razem', {'i': 10}, status=400)
        assert not auth_session.called

def test_move(app, monkeypatch):
    if False:
        i = 10
        return i + 15
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        firetactoe.Game(id='razem', userX=users.get_current_user(), board=9 * ' ', moveX=True).put()
        app.post('/move?g=razem', {'i': 0}, status=200)
        game = ndb.Key('Game', 'razem').get()
        assert game.board == 'X' + 8 * ' '
        auth_session.assert_called_once_with(mock.ANY, method='PATCH', url='http://firebase.com/test-db-url/channels/38razem.json', body='{"winner": null, "userX": "38", "moveX": false, "winningBoard": null, "board": "X        ", "userO": null}', data=None)

def test_delete(app, monkeypatch):
    if False:
        print('Hello World!')
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'access_token': '123'}
        auth_session.return_value = MockResponse(data, http_client.OK)
        firetactoe.Game(id='razem', userX=users.get_current_user()).put()
        app.post('/delete?g=razem', status=200)
        auth_session.assert_called_once_with(mock.ANY, method='DELETE', url='http://firebase.com/test-db-url/channels/38razem.json')
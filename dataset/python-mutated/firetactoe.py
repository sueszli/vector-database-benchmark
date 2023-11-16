"""Tic Tac Toe with the Firebase API"""
import base64
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
import json
import os
import re
import time
import urllib
import flask
from flask import request
from google.appengine.api import app_identity
from google.appengine.api import users
from google.appengine.ext import ndb
from google.auth.transport.requests import AuthorizedSession
import google.auth
_FIREBASE_CONFIG = '_firebase_config.html'
_IDENTITY_ENDPOINT = 'https://identitytoolkit.googleapis.com/google.identity.identitytoolkit.v1.IdentityToolkit'
_FIREBASE_SCOPES = ['https://www.googleapis.com/auth/firebase.database', 'https://www.googleapis.com/auth/userinfo.email']
_X_WIN_PATTERNS = ['XXX......', '...XXX...', '......XXX', 'X..X..X..', '.X..X..X.', '..X..X..X', 'X...X...X', '..X.X.X..']
_O_WIN_PATTERNS = map(lambda s: s.replace('X', 'O'), _X_WIN_PATTERNS)
X_WINS = map(lambda s: re.compile(s), _X_WIN_PATTERNS)
O_WINS = map(lambda s: re.compile(s), _O_WIN_PATTERNS)
app = flask.Flask(__name__)

@lru_cache()
def _get_firebase_db_url():
    if False:
        while True:
            i = 10
    "Grabs the databaseURL from the Firebase config snippet. Regex looks\n    scary, but all it is doing is pulling the 'databaseURL' field from the\n    Firebase javascript snippet"
    regex = re.compile('\\bdatabaseURL\\b.*?["\\\']([^"\\\']+)')
    cwd = os.path.dirname(__file__)
    try:
        with open(os.path.join(cwd, 'templates', _FIREBASE_CONFIG)) as f:
            url = next((regex.search(line) for line in f if regex.search(line)))
    except StopIteration:
        raise ValueError('Error parsing databaseURL. Please copy Firebase web snippet into templates/{}'.format(_FIREBASE_CONFIG))
    return url.group(1)

@lru_cache()
def _get_session():
    if False:
        i = 10
        return i + 15
    'Provides an authed requests session object.'
    (creds, _) = google.auth.default(scopes=[_FIREBASE_SCOPES])
    authed_session = AuthorizedSession(creds)
    return authed_session

def _send_firebase_message(u_id, message=None):
    if False:
        return 10
    'Updates data in firebase. If a message is provided, then it updates\n    the data at /channels/<channel_id> with the message using the PATCH\n    http method. If no message is provided, then the data at this location\n    is deleted using the DELETE http method\n    '
    url = '{}/channels/{}.json'.format(_get_firebase_db_url(), u_id)
    if message:
        return _get_session().patch(url, body=message)
    else:
        return _get_session().delete(url)

def create_custom_token(uid, valid_minutes=60):
    if False:
        while True:
            i = 10
    "Create a secure token for the given id.\n\n    This method is used to create secure custom JWT tokens to be passed to\n    clients. It takes a unique id (uid) that will be used by Firebase's\n    security rules to prevent unauthorized access. In this case, the uid will\n    be the channel id which is a combination of user_id and game_key\n    "
    client_email = app_identity.get_service_account_name()
    now = int(time.time())
    payload = base64.b64encode(json.dumps({'iss': client_email, 'sub': client_email, 'aud': _IDENTITY_ENDPOINT, 'uid': uid, 'iat': now, 'exp': now + valid_minutes * 60}))
    header = base64.b64encode(json.dumps({'typ': 'JWT', 'alg': 'RS256'}))
    to_sign = '{}.{}'.format(header, payload)
    return '{}.{}'.format(to_sign, base64.b64encode(app_identity.sign_blob(to_sign)[1]))

class Game(ndb.Model):
    """All the data we store for a game"""
    userX = ndb.UserProperty()
    userO = ndb.UserProperty()
    board = ndb.StringProperty()
    moveX = ndb.BooleanProperty()
    winner = ndb.StringProperty()
    winning_board = ndb.StringProperty()

    def to_json(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.to_dict()
        d['winningBoard'] = d.pop('winning_board')
        return json.dumps(d, default=lambda user: user.user_id())

    def send_update(self):
        if False:
            return 10
        "Updates Firebase's copy of the board."
        message = self.to_json()
        _send_firebase_message(self.userX.user_id() + self.key.id(), message=message)
        if self.userO:
            _send_firebase_message(self.userO.user_id() + self.key.id(), message=message)

    def _check_win(self):
        if False:
            return 10
        if self.moveX:
            wins = O_WINS
            potential_winner = self.userO.user_id()
        else:
            wins = X_WINS
            potential_winner = self.userX.user_id()
        for win in wins:
            if win.match(self.board):
                self.winner = potential_winner
                self.winning_board = win.pattern
                return
        if ' ' not in self.board:
            self.winner = 'Noone'

    def make_move(self, position, user):
        if False:
            while True:
                i = 10
        if user in (self.userX, self.userO) and self.moveX == (user == self.userX):
            boardList = list(self.board)
            if boardList[position] == ' ':
                boardList[position] = 'X' if self.moveX else 'O'
                self.board = ''.join(boardList)
                self.moveX = not self.moveX
                self._check_win()
                self.put()
                self.send_update()
                return

@app.route('/move', methods=['POST'])
def move():
    if False:
        print('Hello World!')
    game = Game.get_by_id(request.args.get('g'))
    position = int(request.form.get('i'))
    if not (game and 0 <= position <= 8):
        return ('Game not found, or invalid position', 400)
    game.make_move(position, users.get_current_user())
    return ''

@app.route('/delete', methods=['POST'])
def delete():
    if False:
        for i in range(10):
            print('nop')
    game = Game.get_by_id(request.args.get('g'))
    if not game:
        return ('Game not found', 400)
    user = users.get_current_user()
    _send_firebase_message(user.user_id() + game.key.id(), message=None)
    return ''

@app.route('/opened', methods=['POST'])
def opened():
    if False:
        for i in range(10):
            print('nop')
    game = Game.get_by_id(request.args.get('g'))
    if not game:
        return ('Game not found', 400)
    game.send_update()
    return ''

@app.route('/')
def main_page():
    if False:
        print('Hello World!')
    'Renders the main page. When this page is shown, we create a new\n    channel to push asynchronous updates to the client.'
    user = users.get_current_user()
    game_key = request.args.get('g')
    if not game_key:
        game_key = user.user_id()
        game = Game(id=game_key, userX=user, moveX=True, board=' ' * 9)
        game.put()
    else:
        game = Game.get_by_id(game_key)
        if not game:
            return ('No such game', 404)
        if not game.userO:
            game.userO = user
            game.put()
    channel_id = user.user_id() + game_key
    client_auth_token = create_custom_token(channel_id)
    _send_firebase_message(channel_id, message=game.to_json())
    game_link = '{}?g={}'.format(request.base_url, game_key)
    template_values = {'token': client_auth_token, 'channel_id': channel_id, 'me': user.user_id(), 'game_key': game_key, 'game_link': game_link, 'initial_message': urllib.unquote(game.to_json())}
    return flask.render_template('fire_index.html', **template_values)
from urllib.parse import quote
from flask_appbuilder import SQLA
from flask_appbuilder.security.sqla.models import User
from flask_login import current_user
import jwt
from tests.base import FABTestCase

class UserInfoReponseMock:

    def json(self):
        if False:
            for i in range(10):
                print('nop')
        return {'id': '1', 'given_name': 'first-name', 'family_name': 'last-name', 'email': 'user1@fab.org'}

class OAuthRemoteMock:

    def authorize_access_token(self):
        if False:
            return 10
        return {'access_token': 'some-key'}

    def get(self, item):
        if False:
            i = 10
            return i + 15
        if item == 'userinfo':
            return UserInfoReponseMock()

class APICSRFTestCase(FABTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        from flask import Flask
        from flask_wtf import CSRFProtect
        from flask_appbuilder import AppBuilder
        self.app = Flask(__name__)
        self.app.config.from_object('tests.config_oauth')
        self.app.config['WTF_CSRF_ENABLED'] = True
        self.csrf = CSRFProtect(self.app)
        self.db = SQLA(self.app)
        self.appbuilder = AppBuilder(self.app, self.db.session)

    def tearDown(self):
        if False:
            return 10
        self.cleanup()

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        session = self.appbuilder.get_session
        users = session.query(User).filter(User.username.ilike('google%')).all()
        for user in users:
            session.delete(user)
        session.commit()

    def test_oauth_login(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OAuth: Test login\n        '
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        raw_state = {}
        state = jwt.encode(raw_state, 'random_state', algorithm='HS256')
        with self.app.test_client() as client:
            with client.session_transaction() as session_:
                session_['oauth_state'] = 'random_state'
            response = client.get(f'/oauth-authorized/google?state={state}')
            self.assertEqual(current_user.email, 'user1@fab.org')
            self.assertEqual(response.location, '/')

    def test_oauth_login_invalid_state(self):
        if False:
            while True:
                i = 10
        '\n        OAuth: Test login invalid state\n        '
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        raw_state = {}
        state = jwt.encode(raw_state, 'random_state', algorithm='HS256')
        with self.app.test_client() as client:
            with client.session_transaction() as session:
                session['oauth_state'] = 'invalid_state'
            response = client.get(f'/oauth-authorized/google?state={state}')
            self.assertEqual(current_user.is_authenticated, False)
            self.assertEqual(response.location, '/login/')

    def test_oauth_login_unknown_provider(self):
        if False:
            i = 10
            return i + 15
        '\n        OAuth: Test login with unknown provider\n        '
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        raw_state = {}
        state = jwt.encode(raw_state, 'random_state', algorithm='HS256')
        with self.app.test_client() as client:
            with client.session_transaction() as session:
                session['oauth_state'] = 'random_state'
        response = client.get(f'/oauth-authorized/unknown_provider?state={state}')
        self.assertEqual(response.location, '/login/')

    def test_oauth_login_next(self):
        if False:
            print('Hello World!')
        '\n        OAuth: Test login next\n        '
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        raw_state = {'next': ['http://localhost/users/list/']}
        state = jwt.encode(raw_state, 'random_state', algorithm='HS256')
        with self.app.test_client() as client:
            with client.session_transaction() as session:
                session['oauth_state'] = 'random_state'
        response = client.get(f'/oauth-authorized/google?state={state}')
        self.assertEqual(response.location, 'http://localhost/users/list/')

    def test_oauth_login_next_check(self):
        if False:
            while True:
                i = 10
        '\n        OAuth: Test login next check\n        '
        client = self.app.test_client()
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        raw_state = {'next': ['ftp://sample']}
        state = jwt.encode(raw_state, 'random_state', algorithm='HS256')
        with self.app.test_client() as client:
            with client.session_transaction() as session:
                session['oauth_state'] = 'random_state'
        response = client.get(f'/oauth-authorized/google?state={state}')
        self.assertEqual(response.location, '/')

    def test_oauth_next_login_param(self):
        if False:
            print('Hello World!')
        '\n        OAuth: Test next quoted next_url param\n        '
        self.appbuilder.sm.oauth_remotes = {'google': OAuthRemoteMock()}
        next_url = 'http://localhost/data?param1=1&param2=2&param3='
        with self.app.test_client() as client:
            response = client.get(f'/login/?next={quote(next_url)}', follow_redirects=True)
            self.assertTrue(quote(next_url) in response.text)
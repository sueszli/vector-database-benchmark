import os
from datetime import datetime, timezone
from tempfile import NamedTemporaryFile
from unittest import mock
import jwt
import github
from . import Framework
from .GithubIntegration import APP_ID, PRIVATE_KEY, PUBLIC_KEY

class Authentication(Framework.BasicTestCase):

    def testNoAuthentication(self):
        if False:
            while True:
                i = 10
        g = github.Github()
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')

    def testBasicAuthentication(self):
        if False:
            print('Hello World!')
        with self.assertWarns(DeprecationWarning) as warning:
            g = github.Github(self.login.login, self.login.password)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')
        self.assertWarning(warning, 'Arguments login_or_token and password are deprecated, please use auth=github.Auth.Login(...) instead')

    def testOAuthAuthentication(self):
        if False:
            i = 10
            return i + 15
        with self.assertWarns(DeprecationWarning) as warning:
            g = github.Github(self.oauth_token.token)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')
        self.assertWarning(warning, 'Argument login_or_token is deprecated, please use auth=github.Auth.Token(...) instead')

    def testJWTAuthentication(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertWarns(DeprecationWarning) as warning:
            g = github.Github(jwt=self.jwt.token)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')
        self.assertWarning(warning, 'Argument jwt is deprecated, please use auth=github.Auth.AppAuth(...) or auth=github.Auth.AppAuthToken(...) instead')

    def testAppAuthentication(self):
        if False:
            i = 10
            return i + 15
        with self.assertWarns(DeprecationWarning) as warning:
            app_auth = github.AppAuthentication(app_id=self.app_auth.app_id, private_key=self.app_auth.private_key, installation_id=29782936)
            g = github.Github(app_auth=app_auth)
        self.assertEqual(g.get_user('ammarmallik').name, 'Ammar Akbar')
        self.assertWarnings(warning, 'Call to deprecated class AppAuthentication. (Use github.Auth.AppInstallationAuth instead)', 'Argument app_auth is deprecated, please use auth=github.Auth.AppInstallationAuth(...) instead')

    def testLoginAuthentication(self):
        if False:
            i = 10
            return i + 15
        g = github.Github(auth=self.login)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')

    def testTokenAuthentication(self):
        if False:
            print('Hello World!')
        g = github.Github(auth=self.oauth_token)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')

    def testAppAuthTokenAuthentication(self):
        if False:
            i = 10
            return i + 15
        g = github.Github(auth=self.jwt)
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')

    def testAppAuthAuthentication(self):
        if False:
            while True:
                i = 10
        g = github.Github(auth=self.app_auth.get_installation_auth(29782936))
        self.assertEqual(g.get_user('ammarmallik').name, 'Ammar Akbar')

    def assert_requester_args(self, g, expected_requester):
        if False:
            while True:
                i = 10
        expected_args = expected_requester.kwargs
        expected_args.pop('auth')
        auth_args = g._Github__requester.auth.requester.kwargs
        auth_args.pop('auth')
        self.assertEqual(expected_args, auth_args)
        auth_integration_args = g._Github__requester.auth._AppInstallationAuth__integration._GithubIntegration__requester.kwargs
        auth_integration_args.pop('auth')
        self.assertEqual(expected_args, auth_integration_args)

    def testAppAuthAuthenticationWithGithubRequesterArgs(self):
        if False:
            for i in range(10):
                print('nop')
        g = github.Github(auth=self.app_auth.get_installation_auth(29782936), base_url='https://base.net/', timeout=60, user_agent='agent', per_page=100, verify='cert', retry=999, pool_size=10, seconds_between_requests=100, seconds_between_writes=1000)
        self.assert_requester_args(g, g._Github__requester)

    def testAppAuthAuthenticationWithGithubIntegrationRequesterArgs(self):
        if False:
            print('Hello World!')
        gi = github.GithubIntegration(auth=self.app_auth, base_url='https://base.net/', timeout=60, user_agent='agent', per_page=100, verify='cert', retry=999, pool_size=10, seconds_between_requests=100, seconds_between_writes=1000)
        self.assert_requester_args(gi.get_github_for_installation(29782936), gi._GithubIntegration__requester)

    def testAppInstallationAuthAuthentication(self):
        if False:
            for i in range(10):
                print('nop')
        installation_auth = github.Auth.AppInstallationAuth(self.app_auth, 29782936)
        g = github.Github(auth=installation_auth)
        token = installation_auth.token
        self.assertFalse(installation_auth._is_expired)
        self.assertEqual(installation_auth._AppInstallationAuth__installation_authorization.expires_at, datetime(2024, 11, 25, 1, 0, 2, tzinfo=timezone.utc))
        with mock.patch('github.Auth.datetime') as dt:
            dt.now = mock.Mock(return_value=datetime(2024, 11, 25, 0, 59, 3, tzinfo=timezone.utc))
            self.assertFalse(installation_auth._is_expired)
            dt.now = mock.Mock(return_value=datetime(2024, 11, 25, 1, 0, 3, tzinfo=timezone.utc))
            self.assertTrue(installation_auth._is_expired)
            refreshed_token = installation_auth.token
            self.assertNotEqual(refreshed_token, token)
            self.assertFalse(installation_auth._is_expired)
            self.assertEqual(installation_auth._AppInstallationAuth__installation_authorization.expires_at, datetime(2025, 11, 25, 1, 0, 2, tzinfo=timezone.utc))
        self.assertEqual(g.get_user('ammarmallik').name, 'Ammar Akbar')
        self.assertEqual(g.get_repo('PyGithub/PyGithub').full_name, 'PyGithub/PyGithub')

    def testAppInstallationAuthAuthenticationRequesterArgs(self):
        if False:
            print('Hello World!')
        installation_auth = github.Auth.AppInstallationAuth(self.app_auth, 29782936)
        github.Github(auth=installation_auth)

    def testAppUserAuthentication(self):
        if False:
            i = 10
            return i + 15
        client_id = 'removed client id'
        client_secret = 'removed client secret'
        refresh_token = 'removed refresh token'
        g = github.Github()
        app = g.get_oauth_application(client_id, client_secret)
        with mock.patch('github.AccessToken.datetime') as dt:
            dt.now = mock.Mock(return_value=datetime(2023, 6, 7, 12, 0, 0, 123, tzinfo=timezone.utc))
            token = app.refresh_access_token(refresh_token)
        self.assertEqual(token.token, 'fresh access token')
        self.assertEqual(token.type, 'bearer')
        self.assertEqual(token.scope, '')
        self.assertEqual(token.expires_in, 28800)
        self.assertEqual(token.expires_at, datetime(2023, 6, 7, 20, 0, 0, 123, tzinfo=timezone.utc))
        self.assertEqual(token.refresh_token, 'fresh refresh token')
        self.assertEqual(token.refresh_expires_in, 15811200)
        self.assertEqual(token.refresh_expires_at, datetime(2023, 12, 7, 12, 0, 0, 123, tzinfo=timezone.utc))
        auth = app.get_app_user_auth(token)
        with mock.patch('github.Auth.datetime') as dt:
            dt.now = mock.Mock(return_value=datetime(2023, 6, 7, 20, 0, 0, 123, tzinfo=timezone.utc))
            self.assertEqual(auth._is_expired, False)
            self.assertEqual(auth.token, 'fresh access token')
        self.assertEqual(auth.token_type, 'bearer')
        self.assertEqual(auth.refresh_token, 'fresh refresh token')
        with mock.patch('github.Auth.datetime') as dt:
            dt.now = mock.Mock(return_value=datetime(2023, 6, 7, 20, 0, 1, 123, tzinfo=timezone.utc))
            self.assertEqual(auth._is_expired, True)
            self.assertEqual(auth.token, 'another access token')
            self.assertEqual(auth._is_expired, False)
        self.assertEqual(auth.token_type, 'bearer')
        self.assertEqual(auth.refresh_token, 'another refresh token')
        g = github.Github(auth=auth)
        user = g.get_user()
        self.assertEqual(user.login, 'EnricoMi')

    def testNetrcAuth(self):
        if False:
            i = 10
            return i + 15
        with NamedTemporaryFile('wt', delete=False) as tmp:
            tmp.write('machine api.github.com\n')
            tmp.write('login github-user\n')
            tmp.write('password github-password\n')
            tmp.close()
            auth = github.Auth.NetrcAuth()
            with mock.patch.dict(os.environ, {'NETRC': tmp.name}):
                github.Github(auth=auth)
            self.assertEqual(auth.login, 'github-user')
            self.assertEqual(auth.password, 'github-password')
            self.assertEqual(auth.token, 'Z2l0aHViLXVzZXI6Z2l0aHViLXBhc3N3b3Jk')
            self.assertEqual(auth.token_type, 'Basic')

    def testNetrcAuthFails(self):
        if False:
            for i in range(10):
                print('nop')
        with NamedTemporaryFile('wt', delete=False) as tmp:
            tmp.close()
            auth = github.Auth.NetrcAuth()
            with mock.patch.dict(os.environ, {'NETRC': tmp.name}):
                with self.assertRaises(RuntimeError) as exc:
                    github.Github(auth=auth)
                self.assertEqual(exc.exception.args, ('Could not get credentials from netrc for host api.github.com',))

    def testCreateJWT(self):
        if False:
            for i in range(10):
                print('nop')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        with mock.patch('github.Auth.time') as t:
            t.time = mock.Mock(return_value=1550055331.7435968)
            token = auth.create_jwt()
        payload = jwt.decode(token, key=PUBLIC_KEY, algorithms=['RS256'], options={'verify_exp': False})
        self.assertDictEqual(payload, {'iat': 1550055271, 'exp': 1550055631, 'iss': APP_ID})

    def testCreateJWTWithExpiration(self):
        if False:
            while True:
                i = 10
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY, jwt_expiry=120, jwt_issued_at=-30)
        with mock.patch('github.Auth.time') as t:
            t.time = mock.Mock(return_value=1550055331.7435968)
            token = auth.create_jwt(60)
        payload = jwt.decode(token, key=PUBLIC_KEY, algorithms=['RS256'], options={'verify_exp': False})
        self.assertDictEqual(payload, {'iat': 1550055301, 'exp': 1550055391, 'iss': APP_ID})

    def testUserAgent(self):
        if False:
            while True:
                i = 10
        g = github.Github(user_agent='PyGithubTester')
        self.assertEqual(g.get_user('jacquev6').name, 'Vincent Jacques')

    def testAuthorizationHeaderWithLogin(self):
        if False:
            while True:
                i = 10
        g = github.Github(auth=github.Auth.Login('fake_login', 'fake_password'))
        with self.assertRaises(github.GithubException):
            g.get_user().name

    def testAuthorizationHeaderWithToken(self):
        if False:
            for i in range(10):
                print('nop')
        g = github.Github(auth=github.Auth.Token('ZmFrZV9sb2dpbjpmYWtlX3Bhc3N3b3Jk'))
        with self.assertRaises(github.GithubException):
            g.get_user().name
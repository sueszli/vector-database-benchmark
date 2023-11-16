import time
import requests
from urllib3.exceptions import InsecureRequestWarning
import github
from github import Consts
from github.Auth import AppInstallationAuth
from . import Framework
APP_ID = 243473
PRIVATE_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIICXAIBAAKBgQC+5ePolLv6VcWLp2f17g6r6vHl+eoLuodOOfUl8JK+MVmvXbPa\nxDy0SS0pQhwTOMtB0VdSt++elklDCadeokhEoGDQp411o+kiOhzLxfakp/kewf4U\nHJnu4M/A2nHmxXVe2lzYnZvZHX5BM4SJo5PGdr0Ue2JtSXoAtYr6qE9maQIDAQAB\nAoGAFhOJ7sy8jG+837Clcihso+8QuHLVYTPaD+7d7dxLbBlS8NfaQ9Nr3cGUqm/N\nxV9NCjiGa7d/y4w/vrPwGh6UUsA+CvndwDgBd0S3WgIdWvAvHM8wKgNh/GBLLzhT\nBg9BouRUzcT1MjAnkGkWqqCAgN7WrCSUMLt57TNleNWfX90CQQDjvVKTT3pOiavD\n3YcLxwkyeGd0VMvKiS4nV0XXJ97cGXs2GpOGXldstDTnF5AnB6PbukdFLHpsx4sW\nHft3LRWnAkEA1pY15ke08wX6DZVXy7zuQ2izTrWSGySn7B41pn55dlKpttjHeutA\n3BEQKTFvMhBCphr8qST7Wf1SR9FgO0tFbwJAEhHji2yy96hUyKW7IWQZhrem/cP8\np4Va9CQolnnDZRNgg1p4eiDiLu3dhLiJ547joXuWTBbLX/Y1Qvv+B+a74QJBAMCW\nO3WbMZlS6eK6//rIa4ZwN00SxDg8I8FUM45jwBsjgVGrKQz2ilV3sutlhIiH82kk\nm1Iq8LMJGYl/LkDJA10CQBV1C+Xu3ukknr7C4A/4lDCa6Xb27cr1HanY7i89A+Ab\neatdM6f/XVqWp8uPT9RggUV9TjppJobYGT2WrWJMkYw=\n-----END RSA PRIVATE KEY-----\n'
PUBLIC_KEY = '\n-----BEGIN PUBLIC KEY-----\nMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC+5ePolLv6VcWLp2f17g6r6vHl\n+eoLuodOOfUl8JK+MVmvXbPaxDy0SS0pQhwTOMtB0VdSt++elklDCadeokhEoGDQ\np411o+kiOhzLxfakp/kewf4UHJnu4M/A2nHmxXVe2lzYnZvZHX5BM4SJo5PGdr0U\ne2JtSXoAtYr6qE9maQIDAQAB\n-----END PUBLIC KEY-----\n'

class GithubIntegration(Framework.BasicTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.org_installation_id = 30614487
        self.repo_installation_id = 30614431
        self.user_installation_id = 30614431

    def testDeprecatedAppAuth(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertWarns(DeprecationWarning) as warning:
            github_integration = github.GithubIntegration(integration_id=APP_ID, private_key=PRIVATE_KEY)
        installations = github_integration.get_installations()
        self.assertEqual(len(list(installations)), 2)
        self.assertWarning(warning, 'Arguments integration_id, private_key, jwt_expiry, jwt_issued_at and jwt_algorithm are deprecated, please use auth=github.Auth.AppAuth(...) instead')

    def testRequiredAppAuth(self):
        if False:
            print('Hello World!')
        for auth in [self.oauth_token, self.jwt, self.login]:
            with self.assertRaises(AssertionError) as r:
                github.GithubIntegration(auth=auth)
            self.assertEqual(str(r.exception), f'GithubIntegration requires github.Auth.AppAuth authentication, not {type(auth)}')

    def testAppAuth(self):
        if False:
            i = 10
            return i + 15
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installations = github_integration.get_installations()
        self.assertEqual(len(list(installations)), 2)

    def testNoneAppAuth(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError):
            github.GithubIntegration(auth=None)

    def testGetInstallations(self):
        if False:
            for i in range(10):
                print('nop')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installations = github_integration.get_installations()
        self.assertEqual(len(list(installations)), 2)
        self.assertEqual(installations[0].id, self.org_installation_id)
        self.assertEqual(installations[1].id, self.repo_installation_id)

    def testGetGithubForInstallation(self):
        if False:
            return 10
        with self.ignoreWarning(category=InsecureRequestWarning, module='urllib3.connectionpool'):
            kwargs = dict(auth=github.Auth.AppAuth(APP_ID, PRIVATE_KEY), base_url='http://api.github.com', timeout=Consts.DEFAULT_TIMEOUT + 10, user_agent='PyGithub/Python-Test', per_page=Consts.DEFAULT_PER_PAGE + 10, verify=False, retry=3, pool_size=10, seconds_between_requests=100, seconds_between_writes=1000)
            self.assertEqual(kwargs.keys(), github.Requester.Requester.__init__.__annotations__.keys())
            github_integration = github.GithubIntegration(**kwargs)
            g = github_integration.get_github_for_installation(36541767)
            self.assertIsInstance(g._Github__requester.auth, AppInstallationAuth)
            actual = g._Github__requester.kwargs
            kwargs.update(auth=str(AppInstallationAuth))
            actual.update(auth=str(type(actual['auth'])))
            self.assertDictEqual(kwargs, actual)
            repo = g.get_repo('PyGithub/PyGithub')
            self.assertEqual(repo.full_name, 'PyGithub/PyGithub')

    def testGetAccessToken(self):
        if False:
            return 10
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        repo_installation_authorization = github_integration.get_access_token(self.repo_installation_id)
        self.assertEqual(repo_installation_authorization.token, 'ghs_1llwuELtXN5HDOB99XhpcTXdJxbOuF0ZlSmj')
        self.assertDictEqual(repo_installation_authorization.permissions, {'issues': 'read', 'metadata': 'read'})
        self.assertEqual(repo_installation_authorization.repository_selection, 'selected')
        org_installation_authorization = github_integration.get_access_token(self.org_installation_id)
        self.assertEqual(org_installation_authorization.token, 'ghs_V0xygF8yACXSDz5FM65QWV1BT2vtxw0cbgPw')
        org_permissions = {'administration': 'write', 'issues': 'write', 'metadata': 'read', 'organization_administration': 'read'}
        self.assertDictEqual(org_installation_authorization.permissions, org_permissions)
        self.assertEqual(org_installation_authorization.repository_selection, 'selected')
        user_installation_authorization = github_integration.get_access_token(self.user_installation_id)
        self.assertEqual(user_installation_authorization.token, 'ghs_1llwuELtXN5HDOB99XhpcTXdJxbOuF0ZlSmj')
        self.assertDictEqual(user_installation_authorization.permissions, {'issues': 'read', 'metadata': 'read'})
        self.assertEqual(user_installation_authorization.repository_selection, 'selected')

    def testGetUserInstallation(self):
        if False:
            print('Hello World!')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installation = github_integration.get_user_installation(username='ammarmallik')
        self.assertEqual(installation.id, self.user_installation_id)

    def testGetOrgInstallation(self):
        if False:
            while True:
                i = 10
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installation = github_integration.get_org_installation(org='GithubApp-Test-Org')
        self.assertEqual(installation.id, self.org_installation_id)

    def testGetRepoInstallation(self):
        if False:
            print('Hello World!')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installation = github_integration.get_repo_installation(owner='ammarmallik', repo='test-runner')
        self.assertEqual(installation.id, self.repo_installation_id)

    def testGetAppInstallation(self):
        if False:
            i = 10
            return i + 15
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        installation = github_integration.get_app_installation(installation_id=self.org_installation_id)
        self.assertEqual(installation.id, self.org_installation_id)

    def testGetInstallationNotFound(self):
        if False:
            return 10
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.UnknownObjectException) as raisedexp:
            github_integration.get_org_installation(org='GithubApp-Test-Org-404')
        self.assertEqual(raisedexp.exception.status, 404)

    def testGetInstallationWithExpiredJWT(self):
        if False:
            while True:
                i = 10
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.GithubException) as raisedexp:
            github_integration.get_org_installation(org='GithubApp-Test-Org')
        self.assertEqual(raisedexp.exception.status, 401)

    def testGetAccessTokenWithExpiredJWT(self):
        if False:
            print('Hello World!')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.GithubException) as raisedexp:
            github_integration.get_access_token(self.repo_installation_id)
        self.assertEqual(raisedexp.exception.status, 401)

    def testGetAccessTokenForNoInstallation(self):
        if False:
            i = 10
            return i + 15
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.UnknownObjectException) as raisedexp:
            github_integration.get_access_token(40432121)
        self.assertEqual(raisedexp.exception.status, 404)

    def testGetAccessTokenWithInvalidPermissions(self):
        if False:
            i = 10
            return i + 15
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.GithubException) as raisedexp:
            github_integration.get_access_token(self.repo_installation_id, permissions={'test-permissions': 'read'})
        self.assertEqual(raisedexp.exception.status, 422)

    def testGetAccessTokenWithInvalidData(self):
        if False:
            for i in range(10):
                print('nop')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        with self.assertRaises(github.GithubException) as raisedexp:
            github_integration.get_access_token(self.repo_installation_id, permissions='invalid_data')
        self.assertEqual(raisedexp.exception.status, 400)

    def testGetApp(self):
        if False:
            print('Hello World!')
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        github_integration = github.GithubIntegration(auth=auth)
        app = github_integration.get_app()
        self.assertEqual(app.name, 'PyGithubTest')
        self.assertEqual(app.url, '/apps/pygithubtest')
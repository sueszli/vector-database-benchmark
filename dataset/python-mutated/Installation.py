from urllib3.exceptions import InsecureRequestWarning
import github
from github import Consts
from github.Auth import AppAuth, AppInstallationAuth
from . import Framework, GithubIntegration

class Installation(Framework.BasicTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        app_id = 36541767
        private_key = GithubIntegration.PRIVATE_KEY
        self.auth = AppAuth(app_id, private_key)
        self.integration = github.GithubIntegration(auth=self.auth)
        self.installations = list(self.integration.get_installations())

    def testGetRepos(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.installations), 1)
        installation = self.installations[0]
        repos = list(installation.get_repos())
        self.assertEqual(len(repos), 2)
        self.assertListEqual([repo.full_name for repo in repos], ['EnricoMi/sandbox', 'EnricoMi/python'])

    def testGetGithubForInstallation(self):
        if False:
            print('Hello World!')
        with self.ignoreWarning(category=InsecureRequestWarning, module='urllib3.connectionpool'):
            kwargs = dict(auth=AppAuth(319953, GithubIntegration.PRIVATE_KEY), base_url='http://api.github.com', timeout=Consts.DEFAULT_TIMEOUT + 10, user_agent='PyGithub/Python-Test', per_page=Consts.DEFAULT_PER_PAGE + 10, verify=False, retry=3, pool_size=10, seconds_between_requests=100, seconds_between_writes=1000)
            self.assertEqual(kwargs.keys(), github.Requester.Requester.__init__.__annotations__.keys())
            self.integration = github.GithubIntegration(**kwargs)
            installations = list(self.integration.get_installations())
            installation = installations[0]
            g = installation.get_github_for_installation()
            self.assertIsInstance(g._Github__requester.auth, AppInstallationAuth)
            actual = g._Github__requester.kwargs
            kwargs.update(auth=str(AppInstallationAuth))
            actual.update(auth=str(type(actual['auth'])))
            self.assertDictEqual(kwargs, actual)
            repo = g.get_repo('PyGithub/PyGithub')
            self.assertEqual(repo.full_name, 'PyGithub/PyGithub')
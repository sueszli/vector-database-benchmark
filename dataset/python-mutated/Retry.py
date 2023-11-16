import requests
import urllib3
from httpretty import httpretty
import github
from . import Framework
REPO_NAME = 'PyGithub/PyGithub'

class Retry(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        status_forcelist = (500, 502, 504)
        retry = urllib3.Retry(total=3, read=3, connect=3, status_forcelist=status_forcelist)
        Framework.enableRetry(retry)
        super().setUp()

    def testShouldNotRetryWhenStatusNotOnList(self):
        if False:
            print('Hello World!')
        with self.assertRaises(github.GithubException):
            self.g.get_repo(REPO_NAME)
        self.assertEqual(len(httpretty.latest_requests), 1)

    def testReturnsRepoAfter3Retries(self):
        if False:
            while True:
                i = 10
        repository = self.g.get_repo(REPO_NAME)
        self.assertEqual(len(httpretty.latest_requests), 4)
        for request in httpretty.latest_requests:
            self.assertEqual(request.path, '/repos/' + REPO_NAME)
        self.assertIsInstance(repository, github.Repository.Repository)
        self.assertEqual(repository.full_name, REPO_NAME)

    def testReturnsRepoAfter1Retry(self):
        if False:
            return 10
        repository = self.g.get_repo(REPO_NAME)
        self.assertEqual(len(httpretty.latest_requests), 2)
        for request in httpretty.latest_requests:
            self.assertEqual(request.path, '/repos/' + REPO_NAME)
        self.assertIsInstance(repository, github.Repository.Repository)
        self.assertEqual(repository.full_name, REPO_NAME)

    def testRaisesRetryErrorAfterMaxRetries(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(requests.exceptions.RetryError):
            self.g.get_repo('PyGithub/PyGithub')
        self.assertEqual(len(httpretty.latest_requests), 4)
        for request in httpretty.latest_requests:
            self.assertEqual(request.path, '/repos/PyGithub/PyGithub')

    def testReturnsRepoAfterSettingRetryHttp(self):
        if False:
            return 10
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com', retry=0)
        repository = g.get_repo(REPO_NAME)
        self.assertIsInstance(repository, github.Repository.Repository)
        self.assertEqual(repository.full_name, REPO_NAME)
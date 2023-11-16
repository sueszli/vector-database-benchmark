import github
from . import Framework
REPO_NAME = 'PyGithub/PyGithub'

class PoolSize(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        Framework.setPoolSize(20)
        super().setUp()

    def testReturnsRepoAfterSettingPoolSize(self):
        if False:
            for i in range(10):
                print('nop')
        repository = self.g.get_repo(REPO_NAME)
        self.assertIsInstance(repository, github.Repository.Repository)
        self.assertEqual(repository.full_name, REPO_NAME)

    def testReturnsRepoAfterSettingPoolSizeHttp(self):
        if False:
            return 10
        g = github.Github(auth=self.login, base_url='http://my.enterprise.com', pool_size=20)
        repository = g.get_repo(REPO_NAME)
        self.assertIsInstance(repository, github.Repository.Repository)
        self.assertEqual(repository.full_name, REPO_NAME)
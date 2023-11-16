from io import BytesIO as IO
import github
from . import Framework

class Persistence(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.repo = self.g.get_repo('akfish/PyGithub')
        self.dumpedRepo = IO()
        self.g.dump(self.repo, self.dumpedRepo)
        self.dumpedRepo.seek(0)

    def tearDown(self):
        if False:
            return 10
        self.dumpedRepo.close()
        super().tearDown()

    def testLoad(self):
        if False:
            print('Hello World!')
        loadedRepo = self.g.load(self.dumpedRepo)
        self.assertTrue(isinstance(loadedRepo, github.Repository.Repository))
        self.assertTrue(loadedRepo._requester is self.repo._requester)
        self.assertTrue(loadedRepo.owner._requester is self.repo._requester)
        self.assertEqual(loadedRepo.name, 'PyGithub')
        self.assertEqual(loadedRepo.url, 'https://api.github.com/repos/akfish/PyGithub')

    def testLoadAndUpdate(self):
        if False:
            i = 10
            return i + 15
        loadedRepo = self.g.load(self.dumpedRepo)
        self.assertTrue(loadedRepo.update())
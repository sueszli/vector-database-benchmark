from . import Framework

class ConditionalRequestUpdate(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.repo = self.g.get_repo('akfish/PyGithub', lazy=False)

    def testDidNotUpdate(self):
        if False:
            return 10
        self.assertFalse(self.repo.update(), msg='The repo is not changed. But update() != False')

    def testDidUpdate(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.repo.update(), msg='The repo should be changed by now. But update() != True')

    def testUpdateObjectWithoutEtag(self):
        if False:
            i = 10
            return i + 15
        r = self.g.get_repo('jacquev6/PyGithub', lazy=False)
        self.assertTrue(r.update())
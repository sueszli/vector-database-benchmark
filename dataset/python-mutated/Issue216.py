from . import Framework

class Issue216(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.repo = self.g.get_user('openframeworks').get_repo('openFrameworks')
        self.list = self.repo.get_issues()

    def testIteration(self):
        if False:
            return 10
        self.assertEqual(len(list(self.list)), 333)
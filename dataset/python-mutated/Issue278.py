from . import Framework

class Issue278(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.repo = self.g.get_user('openframeworks').get_repo('openFrameworks')
        self.list = self.repo.get_issues()

    def testIteration(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(list(self.list)), 333)
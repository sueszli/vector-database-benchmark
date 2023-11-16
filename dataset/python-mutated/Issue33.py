from . import Framework

class Issue33(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.repo = self.g.get_user('openframeworks').get_repo('openFrameworks')

    def testOpenIssues(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(list(self.repo.get_issues())), 338)

    def testClosedIssues(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(list(self.repo.get_issues(state='closed'))), 950)
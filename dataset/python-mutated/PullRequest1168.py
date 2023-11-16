from . import Framework

class PullRequest1168(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.notifications = self.g.get_repo('PyGithub/PyGithub').get_notifications(all=True)

    def testGetPullRequest(self):
        if False:
            i = 10
            return i + 15
        p = self.notifications[0].get_pull_request()
        self.assertEqual(p.id, 297582636)
        self.assertEqual(p.number, 1171)
        self.assertEqual(p.title, 'Fix small issues for Python 3 compatibility.')

    def testGetIssue(self):
        if False:
            i = 10
            return i + 15
        i = self.notifications[0].get_issue()
        self.assertEqual(i.id, 297582636)
        self.assertEqual(i.number, 1171)
        self.assertEqual(i.title, 'Fix small issues for Python 3 compatibility.')
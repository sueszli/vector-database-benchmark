from . import Framework

class Issue87(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')

    def testCreateIssueWithPercentInTitle(self):
        if False:
            i = 10
            return i + 15
        issue = self.repo.create_issue('Issue with percent % in title created by PyGithub')
        self.assertEqual(issue.number, 99)

    def testCreateIssueWithPercentInBody(self):
        if False:
            i = 10
            return i + 15
        issue = self.repo.create_issue('Issue created by PyGithub', 'Percent % in body')
        self.assertEqual(issue.number, 98)

    def testCreateIssueWithEscapedPercentInTitle(self):
        if False:
            i = 10
            return i + 15
        issue = self.repo.create_issue('Issue with escaped percent %25 in title created by PyGithub')
        self.assertEqual(issue.number, 97)

    def testCreateIssueWithEscapedPercentInBody(self):
        if False:
            i = 10
            return i + 15
        issue = self.repo.create_issue('Issue created by PyGithub', 'Escaped percent %25 in body')
        self.assertEqual(issue.number, 96)
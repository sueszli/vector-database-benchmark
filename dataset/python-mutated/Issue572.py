import github
from . import Framework

class Issue572(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')

    def testIssueAsPullRequest(self):
        if False:
            for i in range(10):
                print('nop')
        issue = self.repo.get_issue(2)
        pull = issue.as_pull_request()
        self.assertEqual(issue.html_url, pull.html_url)
        self.assertTrue(isinstance(pull, github.PullRequest.PullRequest))

    def testPullReqeustAsIssue(self):
        if False:
            return 10
        pull = self.repo.get_pull(2)
        issue = pull.as_issue()
        self.assertEqual(pull.html_url, issue.html_url)
        self.assertTrue(isinstance(issue, github.Issue.Issue))
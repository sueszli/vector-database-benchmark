from datetime import datetime, timezone
from . import Framework

class PullRequestReview(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.repo = self.g.get_repo('PyGithub/PyGithub', lazy=True)
        self.pull = self.repo.get_pull(538)
        self.created_pullreview = self.pull.create_review(self.repo.get_commit('2f0e4e55fe87e38d26efc9aa1346f56abfbd6c52'), 'Some review created by PyGithub')
        self.pullreviews = self.pull.get_reviews()
        self.pullreview = self.pull.get_review(28482091)

    def testDoesNotModifyPullRequest(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.pull.id, 111649703)

    def testDismiss(self):
        if False:
            for i in range(10):
                print('nop')
        self.pullreview.dismiss('with prejudice')
        pr = self.pull.get_review(28482091)
        self.assertEqual(pr.state, 'DISMISSED')

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.pullreview.id, 28482091)
        self.assertEqual(self.pullreview.user.login, 'jzelinskie')
        self.assertEqual(self.pullreview.body, '')
        self.assertEqual(self.pullreview.commit_id, '7a0fcb27b7cd6c346fc3f76216ccb6e0f4ca3bcc')
        self.assertEqual(self.pullreview.state, 'APPROVED')
        self.assertEqual(self.pullreview.html_url, 'https://github.com/PyGithub/PyGithub/pull/538#pullrequestreview-28482091')
        self.assertEqual(self.pullreview.pull_request_url, 'https://api.github.com/repos/PyGithub/PyGithub/pulls/538')
        self.assertEqual(self.pullreview.submitted_at, datetime(2017, 3, 22, 19, 6, 59, tzinfo=timezone.utc))
        self.assertIn(self.created_pullreview.id, [r.id for r in self.pullreviews])
        self.assertEqual(repr(self.pullreview), 'PullRequestReview(user=NamedUser(login="jzelinskie"), id=28482091)')
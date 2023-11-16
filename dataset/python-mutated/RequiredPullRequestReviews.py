from . import Framework

class RequiredPullRequestReviews(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.required_pull_request_reviews = self.g.get_user().get_repo('PyGithub').get_branch('integrations').get_required_pull_request_reviews()

    def testAttributes(self):
        if False:
            return 10
        self.assertTrue(self.required_pull_request_reviews.dismiss_stale_reviews)
        self.assertTrue(self.required_pull_request_reviews.require_code_owner_reviews)
        self.assertEqual(self.required_pull_request_reviews.required_approving_review_count, 3)
        self.assertEqual(self.required_pull_request_reviews.url, 'https://api.github.com/repos/jacquev6/PyGithub/branches/integrations/protection/required_pull_request_reviews')
        self.assertIs(self.required_pull_request_reviews.dismissal_users, None)
        self.assertIs(self.required_pull_request_reviews.dismissal_teams, None)
        self.assertEqual(self.required_pull_request_reviews.__repr__(), 'RequiredPullRequestReviews(url="https://api.github.com/repos/jacquev6/PyGithub/branches/integrations/protection/required_pull_request_reviews", require_code_owner_reviews=True, dismiss_stale_reviews=True)')

    def testOrganizationOwnedTeam(self):
        if False:
            for i in range(10):
                print('nop')
        required_pull_request_reviews = self.g.get_repo('PyGithub/PyGithub', lazy=True).get_branch('integrations').get_required_pull_request_reviews()
        self.assertListKeyEqual(required_pull_request_reviews.dismissal_users, lambda u: u.login, ['jacquev6'])
        self.assertListKeyEqual(required_pull_request_reviews.dismissal_teams, lambda t: t.slug, ['pygithub-owners'])
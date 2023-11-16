import github
from . import Framework

class Branch(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.repo = self.g.get_user().get_repo('PyGithub')
        self.branch = self.repo.get_branch('topic/RewriteWithGeneratedCode')
        self.protected_branch = self.repo.get_branch('integrations')
        self.organization_branch = self.g.get_repo('PyGithub/PyGithub', lazy=True).get_branch('master')

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.branch.name, 'topic/RewriteWithGeneratedCode')
        self.assertEqual(self.branch.commit.sha, '1292bf0e22c796e91cc3d6e24b544aece8c21f2a')
        self.assertEqual(self.branch.protection_url, 'https://api.github.com/repos/jacquev6/PyGithub/branches/topic/RewriteWithGeneratedCode/protection')
        self.assertFalse(self.branch.protected)
        self.assertEqual(repr(self.branch), 'Branch(name="topic/RewriteWithGeneratedCode")')

    def testEditProtection(self):
        if False:
            return 10
        self.protected_branch.edit_protection(strict=True, require_code_owner_reviews=True, required_approving_review_count=2)
        branch_protection = self.protected_branch.get_protection()
        self.assertTrue(branch_protection.required_status_checks.strict)
        self.assertEqual(branch_protection.required_status_checks.contexts, [])
        self.assertTrue(branch_protection.enforce_admins)
        self.assertFalse(branch_protection.required_linear_history)
        self.assertFalse(branch_protection.required_pull_request_reviews.dismiss_stale_reviews)
        self.assertTrue(branch_protection.required_pull_request_reviews.require_code_owner_reviews)
        self.assertEqual(branch_protection.required_pull_request_reviews.required_approving_review_count, 2)

    def testEditProtectionDismissalUsersWithUserOwnedBranch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(github.GithubException) as raisedexp:
            self.protected_branch.edit_protection(dismissal_users=['jacquev6'])
        self.assertEqual(raisedexp.exception.status, 422)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#update-branch-protection', 'message': 'Validation Failed', 'errors': ['Only organization repositories can have users and team restrictions']})

    def testEditProtectionPushRestrictionsWithUserOwnedBranch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(github.GithubException) as raisedexp:
            self.protected_branch.edit_protection(user_push_restrictions=['jacquev6'], team_push_restrictions=[])
        self.assertEqual(raisedexp.exception.status, 422)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#update-branch-protection', 'message': 'Validation Failed', 'errors': ['Only organization repositories can have users and team restrictions']})

    def testEditProtectionPushRestrictionsAndDismissalUser(self):
        if False:
            i = 10
            return i + 15
        self.organization_branch.edit_protection(dismissal_users=['jacquev6'], user_push_restrictions=['jacquev6'])
        branch_protection = self.organization_branch.get_protection()
        self.assertListKeyEqual(branch_protection.required_pull_request_reviews.dismissal_users, lambda u: u.login, ['jacquev6'])
        self.assertListKeyEqual(branch_protection.required_pull_request_reviews.dismissal_teams, lambda u: u.slug, [])
        self.assertListKeyEqual(branch_protection.get_user_push_restrictions(), lambda u: u.login, ['jacquev6'])
        self.assertListKeyEqual(branch_protection.get_team_push_restrictions(), lambda u: u.slug, [])

    def testRemoveProtection(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.protected_branch.protected)
        self.protected_branch.remove_protection()
        protected_branch = self.repo.get_branch('integrations')
        self.assertFalse(protected_branch.protected)
        with self.assertRaises(github.GithubException) as raisedexp:
            protected_branch.get_protection()
        self.assertEqual(raisedexp.exception.status, 404)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#get-branch-protection', 'message': 'Branch not protected'})

    def testEditRequiredStatusChecks(self):
        if False:
            print('Hello World!')
        self.protected_branch.edit_required_status_checks(strict=True)
        required_status_checks = self.protected_branch.get_required_status_checks()
        self.assertTrue(required_status_checks.strict)
        self.assertEqual(required_status_checks.contexts, ['foo/bar'])

    def testRemoveRequiredStatusChecks(self):
        if False:
            while True:
                i = 10
        self.protected_branch.remove_required_status_checks()
        with self.assertRaises(github.GithubException) as raisedexp:
            self.protected_branch.get_required_status_checks()
        self.assertEqual(raisedexp.exception.status, 404)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#get-required-status-checks-of-protected-branch', 'message': 'Required status checks not enabled'})

    def testEditRequiredPullRequestReviews(self):
        if False:
            for i in range(10):
                print('nop')
        self.protected_branch.edit_required_pull_request_reviews(dismiss_stale_reviews=True, required_approving_review_count=2)
        required_pull_request_reviews = self.protected_branch.get_required_pull_request_reviews()
        self.assertTrue(required_pull_request_reviews.dismiss_stale_reviews)
        self.assertTrue(required_pull_request_reviews.require_code_owner_reviews)
        self.assertEqual(required_pull_request_reviews.required_approving_review_count, 2)

    def testEditRequiredPullRequestReviewsWithTooLargeApprovingReviewCount(self):
        if False:
            return 10
        with self.assertRaises(github.GithubException) as raisedexp:
            self.protected_branch.edit_required_pull_request_reviews(required_approving_review_count=9)
        self.assertEqual(raisedexp.exception.status, 422)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#update-pull-request-review-enforcement-of-protected-branch', 'message': 'Invalid request.\n\n9 must be less than or equal to 6.'})

    def testEditRequiredPullRequestReviewsWithUserBranchAndDismissalUsers(self):
        if False:
            return 10
        with self.assertRaises(github.GithubException) as raisedexp:
            self.protected_branch.edit_required_pull_request_reviews(dismissal_users=['jacquev6'])
        self.assertEqual(raisedexp.exception.status, 422)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#update-pull-request-review-enforcement-of-protected-branch', 'message': 'Dismissal restrictions are supported only for repositories owned by an organization.'})

    def testRemoveRequiredPullRequestReviews(self):
        if False:
            print('Hello World!')
        self.protected_branch.remove_required_pull_request_reviews()
        required_pull_request_reviews = self.protected_branch.get_required_pull_request_reviews()
        self.assertFalse(required_pull_request_reviews.dismiss_stale_reviews)
        self.assertFalse(required_pull_request_reviews.require_code_owner_reviews)
        self.assertEqual(required_pull_request_reviews.required_approving_review_count, 1)

    def testAdminEnforcement(self):
        if False:
            for i in range(10):
                print('nop')
        self.protected_branch.remove_admin_enforcement()
        self.assertFalse(self.protected_branch.get_admin_enforcement())
        self.protected_branch.set_admin_enforcement()
        self.assertTrue(self.protected_branch.get_admin_enforcement())

    def testAddUserPushRestrictions(self):
        if False:
            print('Hello World!')
        self.organization_branch.add_user_push_restrictions('sfdye')
        self.assertListKeyEqual(self.organization_branch.get_user_push_restrictions(), lambda u: u.login, ['jacquev6', 'sfdye'])

    def testReplaceUserPushRestrictions(self):
        if False:
            print('Hello World!')
        self.assertListKeyEqual(self.organization_branch.get_user_push_restrictions(), lambda u: u.login, ['jacquev6'])
        self.organization_branch.replace_user_push_restrictions('sfdye')
        self.assertListKeyEqual(self.organization_branch.get_user_push_restrictions(), lambda u: u.login, ['sfdye'])

    def testRemoveUserPushRestrictions(self):
        if False:
            return 10
        self.organization_branch.remove_user_push_restrictions('jacquev6')
        self.assertListKeyEqual(self.organization_branch.get_user_push_restrictions(), lambda u: u.login, ['sfdye'])

    def testAddTeamPushRestrictions(self):
        if False:
            print('Hello World!')
        self.organization_branch.add_team_push_restrictions('pygithub-owners')
        self.assertListKeyEqual(self.organization_branch.get_team_push_restrictions(), lambda t: t.slug, ['pygithub-owners'])

    def testReplaceTeamPushRestrictions(self):
        if False:
            i = 10
            return i + 15
        self.assertListKeyEqual(self.organization_branch.get_team_push_restrictions(), lambda t: t.slug, ['pygithub-owners'])
        self.organization_branch.replace_team_push_restrictions('org-team')
        self.assertListKeyEqual(self.organization_branch.get_team_push_restrictions(), lambda t: t.slug, ['org-team'])

    def testRemoveTeamPushRestrictions(self):
        if False:
            i = 10
            return i + 15
        self.organization_branch.remove_team_push_restrictions('org-team')
        self.assertListKeyEqual(self.organization_branch.get_team_push_restrictions(), lambda t: t.slug, ['pygithub-owners'])

    def testRemovePushRestrictions(self):
        if False:
            for i in range(10):
                print('nop')
        self.organization_branch.remove_push_restrictions()
        with self.assertRaises(github.GithubException) as raisedexp:
            list(self.organization_branch.get_user_push_restrictions())
        self.assertEqual(raisedexp.exception.status, 404)
        self.assertEqual(raisedexp.exception.data, {'documentation_url': 'https://developer.github.com/v3/repos/branches/#list-team-restrictions-of-protected-branch', 'message': 'Push restrictions not enabled'})

    def testGetRequiredSignatures(self):
        if False:
            return 10
        required_signature = self.protected_branch.get_required_signatures()
        assert required_signature

    def testRemoveRequiredSignatures(self):
        if False:
            print('Hello World!')
        self.protected_branch.remove_required_signatures()

    def testAddRequiredSignatures(self):
        if False:
            i = 10
            return i + 15
        self.protected_branch.add_required_signatures()
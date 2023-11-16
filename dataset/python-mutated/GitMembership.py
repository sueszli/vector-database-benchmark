from . import Framework

class GitMembership(Framework.TestCase):

    def testGetMembership(self):
        if False:
            i = 10
            return i + 15
        octocat = self.g.get_user()
        self.assertEqual(octocat.login, 'octocat')
        membership_data = octocat.get_organization_membership('github')
        self.assertEqual(membership_data.user.login, 'octocat')
        self.assertEqual(membership_data.role, 'admin')
        self.assertEqual(membership_data.organization.login, 'github')
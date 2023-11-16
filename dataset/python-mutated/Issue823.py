from . import Framework

class Issue823(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.org = self.g.get_organization('p-society')
        self.team = self.org.get_team(2745783)
        self.pending_invitations = self.team.invitations()

    def testGetPendingInvitationAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        team_url = self.pending_invitations[0].invitation_teams_url
        self.assertEqual(team_url, 'https://api.github.com/organizations/29895434/invitations/6080804/teams')
        inviter = self.pending_invitations[0].inviter.login
        self.assertEqual(inviter, 'palash25')
        role = self.pending_invitations[0].role
        self.assertEqual(role, 'direct_member')
        team_count = self.pending_invitations[0].team_count
        self.assertEqual(team_count, 1)
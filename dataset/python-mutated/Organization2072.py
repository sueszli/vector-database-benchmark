from . import Framework

class Organization2072(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.org = self.g.get_organization('TestOrganization2072')

    def testCancelInvitation(self):
        if False:
            while True:
                i = 10
        self.assertFalse(any([i for i in self.org.invitations() if i.email == 'foo@bar.org']))
        self.org.invite_user(email='foo@bar.org')
        self.assertTrue(any([i for i in self.org.invitations() if i.email == 'foo@bar.org']))
        invitation = [i for i in self.org.invitations() if i.email == 'foo@bar.org'][0]
        self.assertTrue(self.org.cancel_invitation(invitation))
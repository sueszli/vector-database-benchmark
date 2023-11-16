from . import Framework

class OrganizationHasInMembers(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.g.get_user('meneal')
        self.org = self.g.get_organization('RobotWithFeelings')
        self.has_in_members = self.org.has_in_members(self.user)

    def testHasInMembers(self):
        if False:
            return 10
        self.assertTrue(self.has_in_members)
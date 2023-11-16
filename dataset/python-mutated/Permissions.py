from . import Framework

class Permissions(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.userRepo = self.g.get_repo('PyGithub/PyGithub')

    def testUserRepoPermissionAttributes(self):
        if False:
            return 10
        self.assertFalse(self.userRepo.permissions.admin)
        self.assertIs(self.userRepo.permissions.maintain, None)
        self.assertTrue(self.userRepo.permissions.pull)
        self.assertFalse(self.userRepo.permissions.push)
        self.assertIs(self.userRepo.permissions.triage, None)

    def testUserRepoPermissionRepresentation(self):
        if False:
            while True:
                i = 10
        self.assertEqual(repr(self.userRepo.permissions), 'Permissions(triage=None, push=False, pull=True, maintain=None, admin=False)')
from . import Framework

class Issue937(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.user = self.g.get_user()
        self.repo = self.user.get_repo('PyGithub')

    def testCollaboratorsAffiliation(self):
        if False:
            while True:
                i = 10
        collaborators = self.repo.get_collaborators(affiliation='direct')
        self.assertListKeyEqual(collaborators, lambda u: u.login, ['hegde5'])
        with self.assertRaises(AssertionError):
            self.repo.get_collaborators(affiliation='invalid_option')
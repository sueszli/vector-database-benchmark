from . import Framework

class NamedUser1430(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.user = self.g.get_user('ahhda')

    def testGetProjects(self):
        if False:
            i = 10
            return i + 15
        self.assertListKeyBegin(self.user.get_projects(state='all'), lambda e: e.id, [4083095])
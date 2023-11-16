from . import Framework

class Issue131(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.user = self.g.get_user()
        self.repo = self.g.get_user('openmicroscopy').get_repo('ome-documentation')

    def testGetPullWithOrgHeadUser(self):
        if False:
            while True:
                i = 10
        user = self.repo.get_pull(204).head.user
        self.assertEqual(user.login, 'imcf')
        self.assertEqual(user.type, 'Organization')
        self.assertEqual(user.__class__.__name__, 'NamedUser')

    def testGetPullsWithOrgHeadUser(self):
        if False:
            print('Hello World!')
        for pull in self.repo.get_pulls('closed'):
            if pull.number == 204:
                user = pull.head.user
                self.assertEqual(user, None)
                break
        else:
            self.assertTrue(False)
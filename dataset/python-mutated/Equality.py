from . import Framework

class Equality(Framework.TestCase):

    def testUserEquality(self):
        if False:
            i = 10
            return i + 15
        u1 = self.g.get_user('jacquev6')
        u2 = self.g.get_user('jacquev6')
        self.assertEqual(u1, u2)
        self.assertEqual(hash(u1), hash(u2))

    def testUserDifference(self):
        if False:
            return 10
        u1 = self.g.get_user('jacquev6')
        u2 = self.g.get_user('OddBloke')
        self.assertNotEqual(u1, u2)
        self.assertNotEqual(hash(u1), hash(u2))

    def testBranchEquality(self):
        if False:
            while True:
                i = 10
        r = self.g.get_user().get_repo('PyGithub')
        b1 = r.get_branch('develop')
        b2 = r.get_branch('develop')
        self.assertNotEqual(b1._rawData, b2._rawData)
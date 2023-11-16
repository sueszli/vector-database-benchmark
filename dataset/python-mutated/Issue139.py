from . import Framework

class Issue139(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.g.get_user().get_repo('PyGithub').get_issue(139).user

    def testCompletion(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.user._CompletableGithubObject__completed)
        self.assertEqual(self.user.name, 'Ian Ozsvald')
        self.assertTrue(self.user._CompletableGithubObject__completed)
        self.assertEqual(self.user.plan, None)
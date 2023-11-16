from . import Framework

class Issue174(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.repo = self.g.get_repo('twitter/bootstrap')

    def testGetDirContentsWhithHttpRedirect(self):
        if False:
            return 10
        contents = self.repo.get_contents('js/')
        self.assertEqual(len(contents), 15)
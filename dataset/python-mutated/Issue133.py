from . import Framework

class Issue133(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.user = self.g.get_user()

    def testGetPageWithoutInitialArguments(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(self.user.get_followers().get_page(0)), 22)
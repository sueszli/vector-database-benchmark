from . import Framework

class Issue494(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.repo = self.g.get_repo('apache/brooklyn-server', lazy=True)
        self.pull = self.repo.get_pull(465)

    def testRepr(self):
        if False:
            i = 10
            return i + 15
        expected = 'PullRequest(title="Change SetHostnameCustomizer to check if /etc/sysconfig/network exist…", number=465)'
        self.assertEqual(self.pull.__repr__(), expected)
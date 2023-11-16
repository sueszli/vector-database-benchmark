from . import Framework

class SelfHostedActionsRunner(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.g.get_user('ReDASers')
        self.repo = self.user.get_repo('Phishing-Detection')

    def testAttributes(self):
        if False:
            print('Hello World!')
        runner = self.repo.get_self_hosted_runner(2217)
        self.assertEqual(2217, runner.id)
        self.assertEqual('linux', runner.os)
        self.assertEqual('4306125c7c84', runner.name)
        self.assertEqual('offline', runner.status)
        self.assertFalse(runner.busy)
        labels = runner.labels()
        self.assertEqual(3, len(labels))
        self.assertEqual('self-hosted', labels[0]['name'])
        self.assertEqual('X64', labels[1]['name'])
        self.assertEqual('Linux', labels[2]['name'])
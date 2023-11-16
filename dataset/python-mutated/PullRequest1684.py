from . import Framework

class PullRequest1684(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.user = self.g.get_user('ReDASers')
        self.repo = self.user.get_repo('Phishing-Detection')

    def testGetRunners(self):
        if False:
            return 10
        runners = self.repo.get_self_hosted_runners()
        self.assertEqual(19, runners.totalCount)
        runner = runners[0]
        self.assertEqual(1363, runner.id)
        self.assertEqual('windows', runner.os)
        self.assertEqual('0D80B14DC506', runner.name)
        self.assertEqual('offline', runner.status)
        self.assertFalse(runner.busy)
        labels = runner.labels()
        self.assertEqual(3, len(labels))
        self.assertEqual('self-hosted', labels[0]['name'])
        self.assertEqual('Windows', labels[1]['name'])
        self.assertEqual('X64', labels[2]['name'])

    def testDeleteRunnerObject(self):
        if False:
            return 10
        runners = self.repo.get_self_hosted_runners()
        initial_length = runners.totalCount
        runner_to_delete = runners[0]
        result = self.repo.remove_self_hosted_runner(runner_to_delete)
        self.assertTrue(result)
        runners = self.repo.get_self_hosted_runners()
        ids = [runner.id for runner in self.repo.get_self_hosted_runners()]
        self.assertEqual(initial_length - 1, runners.totalCount)
        self.assertNotIn(runner_to_delete.id, ids)

    def testDeleteRunnerId(self):
        if False:
            print('Hello World!')
        ids = [runner.id for runner in self.repo.get_self_hosted_runners()]
        id_to_delete = ids[0]
        result = self.repo.remove_self_hosted_runner(id_to_delete)
        self.assertTrue(result)
        ids = [runner.id for runner in self.repo.get_self_hosted_runners()]
        self.assertNotIn(id_to_delete, ids)
from . import Framework

class PullRequest2408(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.repo = self.g.get_repo('ReDASers/Phishing-Detection')

    def test_get_workflow_runs(self):
        if False:
            for i in range(10):
                print('nop')
        runs = self.repo.get_workflow_runs(head_sha='7aab33f4294ba5141f17bed0aeb1a929f7afc155')
        self.assertEqual(720994709, runs[0].id)
        runs = self.repo.get_workflow_runs(exclude_pull_requests=True)
        self.assertEqual(3519037359, runs[0].id)
from . import Framework

class PullRequest1682(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.repo = self.g.get_repo('ReDASers/Phishing-Detection')

    def test_no_parameters(self):
        if False:
            i = 10
            return i + 15
        runs = self.repo.get_workflow_runs()
        self.assertEqual(313400760, runs[0].id)

    def test_object_parameters(self):
        if False:
            while True:
                i = 10
        branch = self.repo.get_branch('adversary')
        runs = self.repo.get_workflow_runs(branch=branch)
        self.assertEqual(204764033, runs[0].id)
        self.assertEqual(1, runs.totalCount)
        user = self.g.get_user('shahryarabaki')
        runs = self.repo.get_workflow_runs(actor=user)
        self.assertEqual(28372848, runs[0].id)

    def test_string_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        runs = self.repo.get_workflow_runs(actor='xzhou29')
        self.assertEqual(226142695, runs[0].id)
        runs = self.repo.get_workflow_runs(branch='API_Flatten')
        self.assertEqual(287515889, runs[0].id)
        runs = self.repo.get_workflow_runs(event='pull_request')
        self.assertEqual(298867254, runs[0].id)
        runs = self.repo.get_workflow_runs(status='failure')
        self.assertEqual(292080359, runs[0].id)
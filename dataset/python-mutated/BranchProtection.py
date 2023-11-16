from . import Framework

class BranchProtection(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.branch_protection = self.g.get_repo('curvewise-forks/PyGithub').get_branch('master').get_protection()

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.branch_protection.required_status_checks.strict)
        self.assertEqual(self.branch_protection.required_status_checks.contexts, ['build (3.10)'])
        self.assertTrue(self.branch_protection.required_linear_history)
        self.assertEqual(self.branch_protection.url, 'https://api.github.com/repos/curvewise-forks/PyGithub/branches/master/protection')
        self.assertEqual(self.branch_protection.__repr__(), 'BranchProtection(url="https://api.github.com/repos/curvewise-forks/PyGithub/branches/master/protection")')
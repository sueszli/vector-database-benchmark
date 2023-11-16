from . import Framework

class RequiredStatusChecks(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.required_status_checks = self.g.get_user().get_repo('PyGithub').get_branch('integrations').get_required_status_checks()

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.required_status_checks.strict)
        self.assertEqual(self.required_status_checks.contexts, ['foo/bar'])
        self.assertEqual(self.required_status_checks.url, 'https://api.github.com/repos/jacquev6/PyGithub/branches/integrations/protection/required_status_checks')
        self.assertEqual(self.required_status_checks.__repr__(), 'RequiredStatusChecks(url="https://api.github.com/repos/jacquev6/PyGithub/branches/integrations/protection/required_status_checks", strict=True)')
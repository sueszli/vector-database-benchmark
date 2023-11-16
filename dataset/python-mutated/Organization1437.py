from . import Framework

class Organization1437(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.org = self.g.get_organization('PyGithubSampleOrg')

    def testCreateProject(self):
        if False:
            print('Hello World!')
        project = self.org.create_project('Project title', 'This is the body')
        self.assertEqual(project.id, 4115694)
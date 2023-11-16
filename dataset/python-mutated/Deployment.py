from datetime import datetime, timezone
from . import Framework

class Deployment(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.deployment = self.g.get_user().get_repo('PyGithub').get_deployment(263877258)

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.deployment.id, 263877258)
        self.assertEqual(self.deployment.url, 'https://api.github.com/repos/jacquev6/PyGithub/deployments/263877258')
        self.assertEqual(self.deployment.ref, '743f5a58b0bce91c4eab744ff7e39dfca9e6e8a5')
        self.assertEqual(self.deployment.sha, '743f5a58b0bce91c4eab744ff7e39dfca9e6e8a5')
        self.assertEqual(self.deployment.task, 'deploy')
        self.assertEqual(self.deployment.payload, {'test': True})
        self.assertEqual(self.deployment.original_environment, 'test')
        self.assertEqual(self.deployment.environment, 'test')
        self.assertEqual(self.deployment.description, 'Test deployment')
        self.assertEqual(self.deployment.creator.login, 'jacquev6')
        created_at = datetime(2020, 8, 26, 11, 44, 53, tzinfo=timezone.utc)
        self.assertEqual(self.deployment.created_at, created_at)
        self.assertEqual(self.deployment.updated_at, created_at)
        self.assertEqual(self.deployment.transient_environment, True)
        self.assertEqual(self.deployment.production_environment, False)
        self.assertEqual(self.deployment.statuses_url, 'https://api.github.com/repos/jacquev6/PyGithub/deployments/263877258/statuses')
        self.assertEqual(self.deployment.repository_url, 'https://api.github.com/repos/jacquev6/PyGithub')
        self.assertEqual(repr(self.deployment), 'Deployment(url="https://api.github.com/repos/jacquev6/PyGithub/deployments/263877258", id=263877258)')
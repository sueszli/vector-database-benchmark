from datetime import datetime, timezone
from . import Framework

class CommitStatus(Framework.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.statuses = list(self.g.get_user().get_repo('PyGithub').get_commit('1292bf0e22c796e91cc3d6e24b544aece8c21f2a').get_statuses())

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.statuses[0].created_at, datetime(2012, 9, 8, 11, 30, 56, tzinfo=timezone.utc))
        self.assertEqual(self.statuses[0].updated_at, datetime(2012, 9, 8, 11, 30, 56, tzinfo=timezone.utc))
        self.assertEqual(self.statuses[0].creator.login, 'jacquev6')
        self.assertEqual(self.statuses[0].description, 'Status successfuly created by PyGithub')
        self.assertEqual(self.statuses[1].description, None)
        self.assertEqual(self.statuses[0].id, 277040)
        self.assertEqual(self.statuses[0].state, 'success')
        self.assertEqual(self.statuses[1].state, 'pending')
        self.assertEqual(self.statuses[0].context, 'build')
        self.assertEqual(self.statuses[0].target_url, 'https://github.com/jacquev6/PyGithub/issues/67')
        self.assertEqual(self.statuses[1].target_url, None)
        self.assertEqual(repr(self.statuses[0]), 'CommitStatus(state="success", id=277040, context="build")')
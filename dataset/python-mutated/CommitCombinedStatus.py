from datetime import datetime, timezone
from . import Framework

class CommitCombinedStatus(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.combined_status = self.g.get_repo('edx/edx-platform', lazy=True).get_commit('74e70119a23fa3ffb3db19d4590eccfebd72b659').get_combined_status()

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.combined_status.state, 'success')
        self.assertEqual(self.combined_status.statuses[0].url, 'https://api.github.com/repos/edx/edx-platform/statuses/74e70119a23fa3ffb3db19d4590eccfebd72b659')
        self.assertEqual(self.combined_status.statuses[1].id, 390603044)
        self.assertEqual(self.combined_status.statuses[2].state, 'success')
        self.assertEqual(self.combined_status.statuses[3].description, 'Build finished.')
        self.assertEqual(self.combined_status.statuses[4].target_url, 'https://build.testeng.edx.org/job/edx-platform-python-unittests-pr/10504/')
        self.assertEqual(self.combined_status.statuses[4].created_at, datetime(2015, 12, 14, 13, 24, 18, tzinfo=timezone.utc))
        self.assertEqual(self.combined_status.statuses[3].updated_at, datetime(2015, 12, 14, 13, 23, 35, tzinfo=timezone.utc))
        self.assertEqual(self.combined_status.sha, '74e70119a23fa3ffb3db19d4590eccfebd72b659')
        self.assertEqual(self.combined_status.total_count, 6)
        self.assertEqual(self.combined_status.repository.id, 10391073)
        self.assertEqual(self.combined_status.repository.full_name, 'edx/edx-platform')
        self.assertEqual(self.combined_status.commit_url, 'https://api.github.com/repos/edx/edx-platform/commits/74e70119a23fa3ffb3db19d4590eccfebd72b659')
        self.assertEqual(self.combined_status.url, 'https://api.github.com/repos/edx/edx-platform/commits/74e70119a23fa3ffb3db19d4590eccfebd72b659/status')
        self.assertEqual(repr(self.combined_status), 'CommitCombinedStatus(state="success", sha="74e70119a23fa3ffb3db19d4590eccfebd72b659")')
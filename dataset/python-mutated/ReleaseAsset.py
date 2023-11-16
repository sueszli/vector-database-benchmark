from datetime import datetime, timezone
from . import Framework

class ReleaseAsset(Framework.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.release = self.g.get_user().get_repo('PyGithub').get_releases()[0]
        self.asset = self.release.get_assets()[0]

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.release.id, 1210814)
        self.assertEqual(self.asset.id, 16)
        self.assertEqual(self.asset.url, 'https://api.github.com/api/v3/repos/edhollandAL/PyGithub/releases/assets/16')
        self.assertEqual(self.asset.name, 'Archive.zip')
        self.assertEqual(self.asset.label, 'Installation msi & runbook zipped')
        self.assertEqual(self.asset.content_type, 'application/zip')
        self.assertEqual(self.asset.state, 'uploaded')
        self.assertEqual(self.asset.size, 3783)
        self.assertEqual(self.asset.download_count, 2)
        self.assertEqual(self.asset.created_at, datetime(2017, 2, 1, 22, 40, 58, tzinfo=timezone.utc))
        self.assertEqual(self.asset.updated_at, datetime(2017, 2, 1, 22, 44, 58, tzinfo=timezone.utc))
        self.assertEqual(self.asset.browser_download_url, 'https://github.com/edhollandAL/PyGithub/releases/download/v1.25.2/Asset.zip')
        self.assertEqual(self.asset.uploader.login, 'PyGithub')
        self.assertEqual(repr(self.asset), 'GitReleaseAsset(url="https://api.github.com/api/v3/repos/edhollandAL/PyGithub/releases/assets/16")')

    def testDelete(self):
        if False:
            return 10
        self.assertTrue(self.asset.delete_asset())

    def testUpdate(self):
        if False:
            return 10
        new_name = 'updated-name.zip'
        new_label = 'Updated label'
        updated_asset = self.asset.update_asset(new_name, new_label)
        self.assertEqual(updated_asset.name, new_name)
        self.assertNotEqual(self.asset.name, updated_asset.name)
        self.assertEqual(updated_asset.label, new_label)
        self.assertNotEqual(self.asset.label, updated_asset.label)
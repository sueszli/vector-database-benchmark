from datetime import datetime
from dateutil.tz.tz import tzoffset
import github
from . import Framework

class Migration(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.g.get_user()
        self.migration = self.user.get_migrations()[0]

    def testAttributes(self):
        if False:
            return 10
        self.assertEqual(self.migration.id, 25320)
        self.assertEqual(self.migration.owner.login, 'singh811')
        self.assertEqual(self.migration.guid, '608bceae-b790-11e8-8b43-4e3cb0dd56cc')
        self.assertEqual(self.migration.state, 'exported')
        self.assertEqual(self.migration.lock_repositories, False)
        self.assertEqual(self.migration.exclude_attachments, False)
        self.assertEqual(len(self.migration.repositories), 1)
        self.assertEqual(self.migration.repositories[0].name, 'sample-repo')
        self.assertEqual(self.migration.url, 'https://api.github.com/user/migrations/25320')
        self.assertEqual(self.migration.created_at, datetime(2018, 9, 14, 1, 35, 35, tzinfo=tzoffset(None, 19800)))
        self.assertEqual(self.migration.updated_at, datetime(2018, 9, 14, 1, 35, 46, tzinfo=tzoffset(None, 19800)))
        self.assertEqual(repr(self.migration), 'Migration(url="https://api.github.com/user/migrations/25320", state="exported")')

    def testGetArchiveUrlWhenNotExported(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(github.UnknownObjectException, lambda : self.migration.get_archive_url())

    def testGetStatus(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.migration.get_status(), 'exported')

    def testGetArchiveUrlWhenExported(self):
        if False:
            return 10
        self.assertEqual(self.migration.get_archive_url(), 'https://github-cloud.s3.amazonaws.com/migration/25320/24575?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAISTNZFOVBIJMK3TQ%2F20180913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20180913T201100Z&X-Amz-Expires=300&X-Amz-Signature=a0aeb638facd0c78c1ed3ca86022eddbee91e5fe1bb48ee830f54b8b7b305026&X-Amz-SignedHeaders=host&actor_id=41840111&response-content-disposition=filename%3D608bceae-b790-11e8-8b43-4e3cb0dd56cc.tar.gz&response-content-type=application%2Fx-gzip')

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.migration.delete(), None)

    def testGetArchiveUrlWhenDeleted(self):
        if False:
            while True:
                i = 10
        self.assertRaises(github.UnknownObjectException, lambda : self.migration.get_archive_url())

    def testUnlockRepo(self):
        if False:
            return 10
        self.assertEqual(self.migration.unlock_repo('sample-repo'), None)
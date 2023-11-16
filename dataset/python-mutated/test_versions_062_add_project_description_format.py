import hashlib
import sqlalchemy as sa
from twisted.trial import unittest
from buildbot.test.util import migration
from buildbot.util import sautils

class Migration(migration.MigrateTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        return self.setUpMigrateTest()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tearDownMigrateTest()

    def create_tables_thd(self, conn):
        if False:
            print('Hello World!')
        metadata = sa.MetaData()
        metadata.bind = conn
        hash_length = 40
        projects = sautils.Table('projects', metadata, sa.Column('id', sa.Integer, primary_key=True), sa.Column('name', sa.Text, nullable=False), sa.Column('name_hash', sa.String(hash_length), nullable=False), sa.Column('slug', sa.String(50), nullable=False), sa.Column('description', sa.Text, nullable=True))
        projects.create()
        conn.execute(projects.insert(), [{'id': 4, 'name': 'foo', 'description': 'foo_description', 'description_html': None, 'description_format': None, 'slug': 'foo', 'name_hash': hashlib.sha1(b'foo').hexdigest()}])

    def test_update(self):
        if False:
            while True:
                i = 10

        def setup_thd(conn):
            if False:
                return 10
            self.create_tables_thd(conn)

        def verify_thd(conn):
            if False:
                for i in range(10):
                    print('nop')
            metadata = sa.MetaData()
            metadata.bind = conn
            projects = sautils.Table('projects', metadata, autoload=True)
            self.assertIsInstance(projects.c.description_format.type, sa.Text)
            self.assertIsInstance(projects.c.description_html.type, sa.Text)
            q = sa.select([projects.c.name, projects.c.description_format, projects.c.description_html])
            num_rows = 0
            for row in conn.execute(q):
                self.assertIsNone(row.description_format)
                self.assertIsNone(row.description_html)
                num_rows += 1
            self.assertEqual(num_rows, 1)
        return self.do_test_migration('061', '062', setup_thd, verify_thd)
import hashlib
import sqlalchemy as sa
from twisted.trial import unittest
from buildbot.test.util import migration
from buildbot.util import sautils

class Migration(migration.MigrateTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.setUpMigrateTest()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tearDownMigrateTest()

    def create_tables_thd(self, conn):
        if False:
            while True:
                i = 10
        metadata = sa.MetaData()
        metadata.bind = conn
        builders = sautils.Table('builders', metadata, sa.Column('id', sa.Integer, primary_key=True), sa.Column('name', sa.Text, nullable=False), sa.Column('description', sa.Text, nullable=True), sa.Column('name_hash', sa.String(40), nullable=False))
        builders.create()
        conn.execute(builders.insert(), [{'id': 3, 'name': 'foo', 'description': 'foo_description', 'name_hash': hashlib.sha1(b'foo').hexdigest()}])

    def test_update(self):
        if False:
            while True:
                i = 10

        def setup_thd(conn):
            if False:
                print('Hello World!')
            self.create_tables_thd(conn)

        def verify_thd(conn):
            if False:
                for i in range(10):
                    print('nop')
            metadata = sa.MetaData()
            metadata.bind = conn
            projects = sautils.Table('projects', metadata, autoload=True)
            q = sa.select([projects.c.id, projects.c.name, projects.c.name_hash, projects.c.slug, projects.c.description])
            self.assertEqual(conn.execute(q).fetchall(), [])
            builders = sautils.Table('builders', metadata, autoload=True)
            self.assertIsInstance(builders.c.projectid.type, sa.Integer)
            q = sa.select([builders.c.name, builders.c.projectid])
            num_rows = 0
            for row in conn.execute(q):
                self.assertIsNone(row.projectid)
                num_rows += 1
            self.assertEqual(num_rows, 1)
            insp = sa.inspect(conn)
            indexes = insp.get_indexes('projects')
            index_names = [item['name'] for item in indexes]
            self.assertTrue('projects_name_hash' in index_names)
            indexes = insp.get_indexes('builders')
            index_names = [item['name'] for item in indexes]
            self.assertTrue('builders_projectid' in index_names)
        return self.do_test_migration('059', '060', setup_thd, verify_thd)
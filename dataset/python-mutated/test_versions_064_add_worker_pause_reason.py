import sqlalchemy as sa
from twisted.trial import unittest
from buildbot.db.types.json import JsonObject
from buildbot.test.util import migration
from buildbot.util import sautils

class Migration(migration.MigrateTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        return self.setUpMigrateTest()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tearDownMigrateTest()

    def create_tables_thd(self, conn):
        if False:
            print('Hello World!')
        metadata = sa.MetaData()
        metadata.bind = conn
        workers = sautils.Table('workers', metadata, sa.Column('id', sa.Integer, primary_key=True), sa.Column('name', sa.String(50), nullable=False), sa.Column('info', JsonObject, nullable=False), sa.Column('paused', sa.SmallInteger, nullable=False, server_default='0'), sa.Column('graceful', sa.SmallInteger, nullable=False, server_default='0'))
        workers.create()
        conn.execute(workers.insert(), [{'id': 4, 'name': 'worker1', 'info': '{"key": "value"}', 'paused': 0, 'graceful': 0}])

    def test_update(self):
        if False:
            while True:
                i = 10

        def setup_thd(conn):
            if False:
                for i in range(10):
                    print('nop')
            self.create_tables_thd(conn)

        def verify_thd(conn):
            if False:
                return 10
            metadata = sa.MetaData()
            metadata.bind = conn
            workers = sautils.Table('workers', metadata, autoload=True)
            self.assertIsInstance(workers.c.pause_reason.type, sa.Text)
            q = sa.select([workers.c.name, workers.c.pause_reason])
            num_rows = 0
            for row in conn.execute(q):
                self.assertEqual(row.name, 'worker1')
                self.assertIsNone(row.pause_reason)
                num_rows += 1
            self.assertEqual(num_rows, 1)
        return self.do_test_migration('063', '064', setup_thd, verify_thd)
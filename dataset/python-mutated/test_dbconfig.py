from twisted.internet import defer
from twisted.internet import threads
from twisted.trial import unittest
from buildbot.db import dbconfig
from buildbot.test.util import db

class TestDbConfig(db.RealDatabaseMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        yield self.setUpRealDatabase(table_names=['objects', 'object_state'], sqlite_memory=False)
        yield threads.deferToThread(self.createDbConfig)

    def createDbConfig(self):
        if False:
            for i in range(10):
                print('nop')
        self.dbConfig = dbconfig.DbConfig({'db_url': self.db_url}, self.basedir)

    def tearDown(self):
        if False:
            print('Hello World!')
        return self.tearDownRealDatabase()

    def test_basic(self):
        if False:
            return 10

        def thd():
            if False:
                i = 10
                return i + 15
            workersInDB = ['foo', 'bar']
            self.dbConfig.set('workers', workersInDB)
            workers = self.dbConfig.get('workers')
            self.assertEqual(workers, workersInDB)
        return threads.deferToThread(thd)

    def test_default(self):
        if False:
            while True:
                i = 10

        def thd():
            if False:
                while True:
                    i = 10
            workers = self.dbConfig.get('workers', 'default')
            self.assertEqual(workers, 'default')
        return threads.deferToThread(thd)

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')

        def thd():
            if False:
                while True:
                    i = 10
            with self.assertRaises(KeyError):
                self.dbConfig.get('workers')
        return threads.deferToThread(thd)

    def test_init1(self):
        if False:
            return 10
        obj = dbconfig.DbConfig({'db_url': self.db_url}, self.basedir)
        self.assertEqual(obj.db_url, self.db_url)

    def test_init2(self):
        if False:
            print('Hello World!')
        obj = dbconfig.DbConfig({'db': {'db_url': self.db_url}}, self.basedir)
        self.assertEqual(obj.db_url, self.db_url)

    def test_init3(self):
        if False:
            for i in range(10):
                print('nop')
        obj = dbconfig.DbConfig({}, self.basedir)
        self.assertEqual(obj.db_url, 'sqlite:///state.sqlite')

class TestDbConfigNotInitialized(db.RealDatabaseMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        yield self.setUpRealDatabase(table_names=[], sqlite_memory=False)

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            print('Hello World!')
        yield self.tearDownRealDatabase()

    def createDbConfig(self, db_url=None):
        if False:
            i = 10
            return i + 15
        return dbconfig.DbConfig({'db_url': db_url or self.db_url}, self.basedir)

    def test_default(self):
        if False:
            i = 10
            return i + 15

        def thd():
            if False:
                for i in range(10):
                    print('nop')
            db = self.createDbConfig()
            self.assertEqual('foo', db.get('workers', 'foo'))
        return threads.deferToThread(thd)

    def test_error(self):
        if False:
            for i in range(10):
                print('nop')

        def thd():
            if False:
                while True:
                    i = 10
            db = self.createDbConfig()
            with self.assertRaises(KeyError):
                db.get('workers')
        return threads.deferToThread(thd)

    def test_bad_url(self):
        if False:
            while True:
                i = 10

        def thd():
            if False:
                i = 10
                return i + 15
            db = self.createDbConfig('garbage://')
            with self.assertRaises(KeyError):
                db.get('workers')
        return threads.deferToThread(thd)

    def test_bad_url2(self):
        if False:
            print('Hello World!')

        def thd():
            if False:
                for i in range(10):
                    print('nop')
            db = self.createDbConfig('trash')
            with self.assertRaises(KeyError):
                db.get('workers')
        return threads.deferToThread(thd)

    def test_bad_url3(self):
        if False:
            print('Hello World!')

        def thd():
            if False:
                for i in range(10):
                    print('nop')
            db = self.createDbConfig('sqlite://bad')
            with self.assertRaises(KeyError):
                db.get('workers')
        return threads.deferToThread(thd)
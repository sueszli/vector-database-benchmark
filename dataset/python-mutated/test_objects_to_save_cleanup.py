import unittest
from pony.orm import *
from pony.orm.tests import setup_database, teardown_database

class EntityStatusTestCase(object):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        db = cls.db = Database()

        class TestPost(db.Entity):
            category = Optional('TestCategory')
            name = Optional(str, default='Noname')

        class TestCategory(db.Entity):
            posts = Set(TestPost)
        setup_database(db)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        teardown_database(cls.db)

    def make_flush(self, obj=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @db_session
    def test_delete_updated(self):
        if False:
            return 10
        p = self.db.TestPost()
        self.make_flush(p)
        p.name = 'Pony'
        self.assertEqual(p._status_, 'modified')
        self.make_flush(p)
        self.assertEqual(p._status_, 'updated')
        p.delete()
        self.assertEqual(p._status_, 'marked_to_delete')
        self.make_flush(p)
        self.assertEqual(p._status_, 'deleted')

    @db_session
    def test_delete_inserted(self):
        if False:
            while True:
                i = 10
        p = self.db.TestPost()
        self.assertEqual(p._status_, 'created')
        self.make_flush(p)
        self.assertEqual(p._status_, 'inserted')
        p.delete()

    @db_session
    def test_cancelled(self):
        if False:
            print('Hello World!')
        p = self.db.TestPost()
        self.assertEqual(p._status_, 'created')
        p.delete()
        self.assertEqual(p._status_, 'cancelled')
        self.make_flush(p)
        self.assertEqual(p._status_, 'cancelled')

class EntityStatusTestCase_ObjectFlush(EntityStatusTestCase, unittest.TestCase):

    def make_flush(self, obj=None):
        if False:
            print('Hello World!')
        obj.flush()

class EntityStatusTestCase_FullFlush(EntityStatusTestCase, unittest.TestCase):

    def make_flush(self, obj=None):
        if False:
            for i in range(10):
                print('nop')
        flush()
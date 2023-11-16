import apsw
import datetime
from playhouse.apsw_ext import *
from .base import ModelTestCase
from .base import TestModel
database = APSWDatabase(':memory:')

class User(TestModel):
    username = TextField()

class Message(TestModel):
    user = ForeignKeyField(User)
    message = TextField()
    pub_date = DateTimeField()
    published = BooleanField()

class VTSource(object):

    def Create(self, db, modulename, dbname, tablename, *args):
        if False:
            return 10
        schema = 'CREATE TABLE x(value)'
        return (schema, VTable())
    Connect = Create

class VTable(object):

    def BestIndex(self, *args):
        if False:
            return 10
        return

    def Open(self):
        if False:
            for i in range(10):
                print('nop')
        return VTCursor()

    def Disconnect(self):
        if False:
            i = 10
            return i + 15
        pass
    Destroy = Disconnect

class VTCursor(object):

    def Filter(self, *a):
        if False:
            return 10
        self.val = 0

    def Eof(self):
        if False:
            i = 10
            return i + 15
        return False

    def Rowid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.val

    def Column(self, col):
        if False:
            i = 10
            return i + 15
        return self.val

    def Next(self):
        if False:
            return 10
        self.val += 1

    def Close(self):
        if False:
            while True:
                i = 10
        pass

class TestAPSWExtension(ModelTestCase):
    database = database
    requires = [User, Message]

    def test_db_register_module(self):
        if False:
            return 10
        database.register_module('series', VTSource())
        database.execute_sql('create virtual table foo using series()')
        curs = database.execute_sql('select * from foo limit 5;')
        self.assertEqual([v for (v,) in curs], [0, 1, 2, 3, 4])
        database.unregister_module('series')

    def test_db_register_function(self):
        if False:
            while True:
                i = 10

        @database.func()
        def title(s):
            if False:
                return 10
            return s.title()
        curs = self.database.execute_sql('SELECT title(?)', ('heLLo',))
        self.assertEqual(curs.fetchone()[0], 'Hello')

    def test_db_register_aggregate(self):
        if False:
            print('Hello World!')

        @database.aggregate()
        class First(object):

            def __init__(self):
                if False:
                    return 10
                self._value = None

            def step(self, value):
                if False:
                    return 10
                if self._value is None:
                    self._value = value

            def finalize(self):
                if False:
                    while True:
                        i = 10
                return self._value
        with database.atomic():
            for i in range(10):
                User.create(username='u%s' % i)
        query = User.select(fn.First(User.username)).order_by(User.username)
        self.assertEqual(query.scalar(), 'u0')

    def test_db_register_collation(self):
        if False:
            for i in range(10):
                print('nop')

        @database.collation()
        def reverse(lhs, rhs):
            if False:
                return 10
            (lhs, rhs) = (lhs.lower(), rhs.lower())
            if lhs < rhs:
                return 1
            return -1 if rhs > lhs else 0
        with database.atomic():
            for i in range(3):
                User.create(username='u%s' % i)
        query = User.select(User.username).order_by(User.username.collate('reverse'))
        self.assertEqual([u.username for u in query], ['u2', 'u1', 'u0'])

    def test_db_pragmas(self):
        if False:
            print('Hello World!')
        test_db = APSWDatabase(':memory:', pragmas=(('cache_size', '1337'),))
        test_db.connect()
        cs = test_db.execute_sql('PRAGMA cache_size;').fetchone()[0]
        self.assertEqual(cs, 1337)

    def test_select_insert(self):
        if False:
            for i in range(10):
                print('nop')
        for user in ('u1', 'u2', 'u3'):
            User.create(username=user)
        self.assertEqual([x.username for x in User.select()], ['u1', 'u2', 'u3'])
        dt = datetime.datetime(2012, 1, 1, 11, 11, 11)
        Message.create(user=User.get(User.username == 'u1'), message='herps', pub_date=dt, published=True)
        Message.create(user=User.get(User.username == 'u2'), message='derps', pub_date=dt, published=False)
        m1 = Message.get(Message.message == 'herps')
        self.assertEqual(m1.user.username, 'u1')
        self.assertEqual(m1.pub_date, dt)
        self.assertEqual(m1.published, True)
        m2 = Message.get(Message.message == 'derps')
        self.assertEqual(m2.user.username, 'u2')
        self.assertEqual(m2.pub_date, dt)
        self.assertEqual(m2.published, False)

    def test_update_delete(self):
        if False:
            i = 10
            return i + 15
        u1 = User.create(username='u1')
        u2 = User.create(username='u2')
        u1.username = 'u1-modified'
        u1.save()
        self.assertEqual(User.select().count(), 2)
        self.assertEqual(User.get(User.username == 'u1-modified').id, u1.id)
        u1.delete_instance()
        self.assertEqual(User.select().count(), 1)

    def test_transaction_handling(self):
        if False:
            i = 10
            return i + 15
        dt = datetime.datetime(2012, 1, 1, 11, 11, 11)

        def do_ctx_mgr_error():
            if False:
                while True:
                    i = 10
            with self.database.transaction():
                User.create(username='u1')
                raise ValueError
        self.assertRaises(ValueError, do_ctx_mgr_error)
        self.assertEqual(User.select().count(), 0)

        def do_ctx_mgr_success():
            if False:
                return 10
            with self.database.transaction():
                u = User.create(username='test')
                Message.create(message='testing', user=u, pub_date=dt, published=1)
        do_ctx_mgr_success()
        self.assertEqual(User.select().count(), 1)
        self.assertEqual(Message.select().count(), 1)

        def create_error():
            if False:
                while True:
                    i = 10
            with self.database.atomic():
                u = User.create(username='test')
                Message.create(message='testing', user=u, pub_date=dt, published=1)
                raise ValueError
        self.assertRaises(ValueError, create_error)
        self.assertEqual(User.select().count(), 1)

        def create_success():
            if False:
                while True:
                    i = 10
            with self.database.atomic():
                u = User.create(username='test')
                Message.create(message='testing', user=u, pub_date=dt, published=1)
        create_success()
        self.assertEqual(User.select().count(), 2)
        self.assertEqual(Message.select().count(), 2)

    def test_exists_regression(self):
        if False:
            i = 10
            return i + 15
        User.create(username='u1')
        self.assertTrue(User.select().where(User.username == 'u1').exists())
        self.assertFalse(User.select().where(User.username == 'ux').exists())
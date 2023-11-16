from itertools import permutations
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import platform
import re
import threading
from peewee import *
from peewee import Database
from peewee import FIELD
from peewee import attrdict
from peewee import sort_models
from .base import BaseTestCase
from .base import DatabaseTestCase
from .base import IS_CRDB
from .base import IS_MYSQL
from .base import IS_POSTGRESQL
from .base import IS_SQLITE
from .base import ModelTestCase
from .base import TestModel
from .base import db
from .base import db_loader
from .base import get_in_memory_db
from .base import new_connection
from .base import requires_models
from .base import requires_postgresql
from .base_models import Category
from .base_models import Tweet
from .base_models import User

class TestDatabase(DatabaseTestCase):
    database = db_loader('sqlite3')

    def test_pragmas(self):
        if False:
            while True:
                i = 10
        self.database.cache_size = -2048
        self.assertEqual(self.database.cache_size, -2048)
        self.database.cache_size = -4096
        self.assertEqual(self.database.cache_size, -4096)
        self.database.foreign_keys = 'on'
        self.assertEqual(self.database.foreign_keys, 1)
        self.database.foreign_keys = 'off'
        self.assertEqual(self.database.foreign_keys, 0)

    def test_appid_user_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.database.application_id, 0)
        self.assertEqual(self.database.user_version, 0)
        self.database.application_id = 1
        self.database.user_version = 2
        self.assertEqual(self.database.application_id, 1)
        self.assertEqual(self.database.user_version, 2)
        self.assertTrue(self.database.close())
        self.assertTrue(self.database.connect())
        self.assertEqual(self.database.application_id, 1)
        self.assertEqual(self.database.user_version, 2)

    def test_timeout_semantics(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.database.timeout, 5)
        self.assertEqual(self.database.pragma('busy_timeout'), 5000)
        self.database.timeout = 2.5
        self.assertEqual(self.database.timeout, 2.5)
        self.assertEqual(self.database.pragma('busy_timeout'), 2500)
        self.database.close()
        self.database.connect()
        self.assertEqual(self.database.timeout, 2.5)
        self.assertEqual(self.database.pragma('busy_timeout'), 2500)

    def test_pragmas_deferred(self):
        if False:
            while True:
                i = 10
        pragmas = (('journal_mode', 'wal'),)
        db = SqliteDatabase(None, pragmas=pragmas)
        self.assertEqual(db._pragmas, pragmas)
        db.init(':memory:')
        self.assertEqual(db._pragmas, pragmas)
        db = SqliteDatabase(None)
        self.assertEqual(db._pragmas, ())
        db.init(':memory:', pragmas=pragmas)
        self.assertEqual(db._pragmas, pragmas)
        db.init(':memory:', pragmas=())
        self.assertEqual(db._pragmas, ())
        db = SqliteDatabase(None, pragmas=pragmas)
        db.init(':memory:', pragmas=(('cache_size', -8000),))
        self.assertEqual(db._pragmas, (('cache_size', -8000),))

    def test_pragmas_as_dict(self):
        if False:
            while True:
                i = 10
        pragmas = {'journal_mode': 'wal'}
        pragma_list = [('journal_mode', 'wal')]
        db = SqliteDatabase(':memory:', pragmas=pragmas)
        self.assertEqual(db._pragmas, pragma_list)
        db = SqliteDatabase(None, pragmas=pragmas)
        self.assertEqual(db._pragmas, pragma_list)
        db.init(':memory:')
        self.assertEqual(db._pragmas, pragma_list)
        db.init(':memory:', pragmas={})
        self.assertEqual(db._pragmas, [])

    def test_pragmas_permanent(self):
        if False:
            while True:
                i = 10
        db = SqliteDatabase(':memory:')
        db.execute_sql('pragma foreign_keys=0')
        self.assertEqual(db.foreign_keys, 0)
        db.pragma('foreign_keys', 1, True)
        self.assertEqual(db.foreign_keys, 1)
        db.close()
        db.connect()
        self.assertEqual(db.foreign_keys, 1)

    def test_context_settings(self):
        if False:
            i = 10
            return i + 15

        class TestDatabase(Database):
            field_types = {'BIGINT': 'TEST_BIGINT', 'TEXT': 'TEST_TEXT'}
            operations = {'LIKE': '~', 'NEW': '->>'}
            param = '$'
        test_db = TestDatabase(None)
        state = test_db.get_sql_context().state
        self.assertEqual(state.field_types['BIGINT'], 'TEST_BIGINT')
        self.assertEqual(state.field_types['TEXT'], 'TEST_TEXT')
        self.assertEqual(state.field_types['INT'], FIELD.INT)
        self.assertEqual(state.field_types['VARCHAR'], FIELD.VARCHAR)
        self.assertEqual(state.operations['LIKE'], '~')
        self.assertEqual(state.operations['NEW'], '->>')
        self.assertEqual(state.operations['ILIKE'], 'ILIKE')
        self.assertEqual(state.param, '$')
        self.assertEqual(state.quote, '""')
        test_db2 = TestDatabase(None, field_types={'BIGINT': 'XXX_BIGINT', 'INT': 'XXX_INT'})
        state = test_db2.get_sql_context().state
        self.assertEqual(state.field_types['BIGINT'], 'XXX_BIGINT')
        self.assertEqual(state.field_types['TEXT'], 'TEST_TEXT')
        self.assertEqual(state.field_types['INT'], 'XXX_INT')
        self.assertEqual(state.field_types['VARCHAR'], FIELD.VARCHAR)

    def test_connection_state(self):
        if False:
            return 10
        conn = self.database.connection()
        self.assertFalse(self.database.is_closed())
        self.database.close()
        self.assertTrue(self.database.is_closed())
        conn = self.database.connection()
        self.assertFalse(self.database.is_closed())

    def test_db_context_manager(self):
        if False:
            print('Hello World!')
        self.database.close()
        self.assertTrue(self.database.is_closed())
        with self.database:
            self.assertFalse(self.database.is_closed())
        self.assertTrue(self.database.is_closed())
        self.database.connect()
        self.assertFalse(self.database.is_closed())
        with self.database:
            self.assertFalse(self.database.is_closed())
        self.assertTrue(self.database.is_closed())

    def test_connection_initialization(self):
        if False:
            return 10
        state = {'count': 0}

        class TestDatabase(SqliteDatabase):

            def _initialize_connection(self, conn):
                if False:
                    return 10
                state['count'] += 1
        db = TestDatabase(':memory:')
        self.assertEqual(state['count'], 0)
        conn = db.connection()
        self.assertEqual(state['count'], 1)
        conn = db.connection()
        self.assertEqual(state['count'], 1)

    def test_connect_semantics(self):
        if False:
            print('Hello World!')
        state = {'count': 0}

        class TestDatabase(SqliteDatabase):

            def _initialize_connection(self, conn):
                if False:
                    for i in range(10):
                        print('nop')
                state['count'] += 1
        db = TestDatabase(':memory:')
        db.connect()
        self.assertEqual(state['count'], 1)
        self.assertRaises(OperationalError, db.connect)
        self.assertEqual(state['count'], 1)
        self.assertFalse(db.connect(reuse_if_open=True))
        self.assertEqual(state['count'], 1)
        with db:
            self.assertEqual(state['count'], 1)
            self.assertFalse(db.is_closed())
        self.assertTrue(db.is_closed())
        with db:
            self.assertEqual(state['count'], 2)

    def test_execute_sql(self):
        if False:
            print('Hello World!')
        self.database.execute_sql('CREATE TABLE register (val INTEGER);')
        self.database.execute_sql('INSERT INTO register (val) VALUES (?), (?)', (1337, 31337))
        cursor = self.database.execute_sql('SELECT val FROM register ORDER BY val')
        self.assertEqual(cursor.fetchall(), [(1337,), (31337,)])
        self.database.execute_sql('DROP TABLE register;')

    def test_bind_helpers(self):
        if False:
            print('Hello World!')
        db = get_in_memory_db()
        alt_db = get_in_memory_db()

        class Base(Model):

            class Meta:
                database = db

        class A(Base):
            a = TextField()

        class B(Base):
            b = TextField()
        db.create_tables([A, B])
        with alt_db.bind_ctx([A]):
            self.assertFalse(A.table_exists())
            self.assertTrue(B.table_exists())
        self.assertTrue(A.table_exists())
        self.assertTrue(B.table_exists())
        alt_db.bind([A])
        self.assertFalse(A.table_exists())
        self.assertTrue(B.table_exists())
        db.close()
        alt_db.close()

    def test_bind_regression(self):
        if False:
            for i in range(10):
                print('nop')

        class Base(Model):

            class Meta:
                database = None

        class A(Base):
            pass

        class B(Base):
            pass

        class AB(Base):
            a = ForeignKeyField(A)
            b = ForeignKeyField(B)
        self.assertTrue(A._meta.database is None)
        db = get_in_memory_db()
        with db.bind_ctx([A, B]):
            self.assertEqual(A._meta.database, db)
            self.assertEqual(B._meta.database, db)
            self.assertEqual(AB._meta.database, db)
        self.assertTrue(A._meta.database is None)
        self.assertTrue(B._meta.database is None)
        self.assertTrue(AB._meta.database is None)

        class C(Base):
            a = ForeignKeyField(A)
        with db.bind_ctx([C], bind_refs=False):
            self.assertEqual(C._meta.database, db)
            self.assertTrue(A._meta.database is None)
        self.assertTrue(C._meta.database is None)
        self.assertTrue(A._meta.database is None)

    def test_batch_commit(self):
        if False:
            print('Hello World!')

        class PatchCommitDatabase(SqliteDatabase):
            commits = 0

            def begin(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def commit(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.commits += 1
        db = PatchCommitDatabase(':memory:')

        def assertBatches(n_objs, batch_size, n_commits):
            if False:
                for i in range(10):
                    print('nop')
            accum = []
            source = range(n_objs)
            db.commits = 0
            for item in db.batch_commit(source, batch_size):
                accum.append(item)
            self.assertEqual(accum, list(range(n_objs)))
            self.assertEqual(db.commits, n_commits)
        assertBatches(12, 1, 12)
        assertBatches(12, 2, 6)
        assertBatches(12, 3, 4)
        assertBatches(12, 4, 3)
        assertBatches(12, 5, 3)
        assertBatches(12, 6, 2)
        assertBatches(12, 7, 2)
        assertBatches(12, 11, 2)
        assertBatches(12, 12, 1)
        assertBatches(12, 13, 1)

    def test_server_version(self):
        if False:
            while True:
                i = 10

        class FakeDatabase(Database):
            server_version = None

            def _connect(self):
                if False:
                    while True:
                        i = 10
                return 1

            def _close(self, conn):
                if False:
                    print('Hello World!')
                pass

            def _set_server_version(self, conn):
                if False:
                    print('Hello World!')
                self.server_version = (1, 33, 7)
        db = FakeDatabase(':memory:')
        self.assertTrue(db.server_version is None)
        db.connect()
        self.assertEqual(db.server_version, (1, 33, 7))
        db.close()
        self.assertEqual(db.server_version, (1, 33, 7))
        db.server_version = (1, 2, 3)
        db.connect()
        self.assertEqual(db.server_version, (1, 2, 3))
        db.close()

    def test_explicit_connect(self):
        if False:
            return 10
        db = get_in_memory_db(autoconnect=False)
        self.assertRaises(InterfaceError, db.execute_sql, 'pragma cache_size')
        with db:
            db.execute_sql('pragma cache_size')
        self.assertRaises(InterfaceError, db.cursor)

class TestThreadSafety(ModelTestCase):
    if IS_SQLITE and platform.machine() not in ('i386', 'i686', 'x86_64'):
        database = new_connection(timeout=60)
    nthreads = 4
    nrows = 10
    requires = [User]

    def test_multiple_writers(self):
        if False:
            print('Hello World!')

        def create_users(idx):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(idx * self.nrows, (idx + 1) * self.nrows):
                User.create(username='u%d' % i)
        threads = []
        for i in range(self.nthreads):
            threads.append(threading.Thread(target=create_users, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(User.select().count(), self.nrows * self.nthreads)

    def test_multiple_readers(self):
        if False:
            return 10
        data = Queue()

        def read_user_count(n):
            if False:
                while True:
                    i = 10
            for i in range(n):
                data.put(User.select().count())
        threads = []
        for i in range(self.nthreads):
            threads.append(threading.Thread(target=read_user_count, args=(self.nrows,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(data.qsize(), self.nrows * self.nthreads)

class TestDeferredDatabase(BaseTestCase):

    def test_deferred_database(self):
        if False:
            return 10
        deferred_db = SqliteDatabase(None)
        self.assertTrue(deferred_db.deferred)

        class DeferredModel(Model):

            class Meta:
                database = deferred_db
        self.assertRaises(Exception, deferred_db.connect)
        query = DeferredModel.select()
        self.assertRaises(Exception, query.execute)
        deferred_db.init(':memory:')
        self.assertFalse(deferred_db.deferred)
        conn = deferred_db.connect()
        self.assertFalse(deferred_db.is_closed())
        DeferredModel._schema.create_all()
        self.assertEqual(list(DeferredModel.select()), [])
        deferred_db.init(None)
        self.assertTrue(deferred_db.deferred)
        self.assertTrue(deferred_db.is_closed())

class CatToy(TestModel):
    description = TextField()

    class Meta:
        schema = 'huey'

@requires_postgresql
class TestSchemaNamespace(ModelTestCase):
    requires = [CatToy]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with self.database:
            self.execute('CREATE SCHEMA huey;')
        super(TestSchemaNamespace, self).setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestSchemaNamespace, self).tearDown()
        with self.database:
            self.execute('DROP SCHEMA huey;')

    def test_schema(self):
        if False:
            while True:
                i = 10
        toy = CatToy.create(description='fur mouse')
        toy_db = CatToy.select().where(CatToy.id == toy.id).get()
        self.assertEqual(toy.id, toy_db.id)
        self.assertEqual(toy.description, toy_db.description)

class TestSqliteIsolation(ModelTestCase):
    database = db_loader('sqlite3')
    requires = [User]

    def test_sqlite_isolation(self):
        if False:
            i = 10
            return i + 15
        for username in ('u1', 'u2', 'u3'):
            User.create(username=username)
        new_db = db_loader('sqlite3')
        curs = new_db.execute_sql('SELECT COUNT(*) FROM users')
        self.assertEqual(curs.fetchone()[0], 3)
        self.assertEqual(User.select().count(), 3)
        self.assertEqual(User.delete().execute(), 3)
        with self.database.atomic():
            User.create(username='u4')
            User.create(username='u5')
            curs = new_db.execute_sql('SELECT COUNT(*) FROM users')
            self.assertEqual(curs.fetchone()[0], 0)
            new_db2 = db_loader('sqlite3')
            curs = new_db2.execute_sql('SELECT COUNT(*) FROM users')
            self.assertEqual(curs.fetchone()[0], 0)
            self.assertEqual(User.select().count(), 2)
        curs = new_db.execute_sql('SELECT COUNT(*) FROM users')
        self.assertEqual(curs.fetchone()[0], 2)

class UniqueModel(TestModel):
    name = CharField(unique=True)

class IndexedModel(TestModel):
    first = CharField()
    last = CharField()
    dob = DateField()

    class Meta:
        indexes = ((('first', 'last', 'dob'), True), (('first', 'last'), False))

class Note(TestModel):
    content = TextField()
    ts = DateTimeField()
    status = IntegerField()

    class Meta:
        table_name = 'notes'

class Person(TestModel):
    first = CharField()
    last = CharField()
    email = CharField()

    class Meta:
        indexes = ((('last', 'first'), False),)

class TestIntrospection(ModelTestCase):
    requires = [Category, User, UniqueModel, IndexedModel, Person]

    def test_table_exists(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.database.table_exists(User._meta.table_name))
        self.assertFalse(self.database.table_exists('nuggies'))
        self.assertTrue(self.database.table_exists(User))

        class X(TestModel):
            pass
        self.assertFalse(self.database.table_exists(X))

    def test_get_tables(self):
        if False:
            while True:
                i = 10
        tables = self.database.get_tables()
        required = set((m._meta.table_name for m in self.requires))
        self.assertTrue(required.issubset(set(tables)))
        UniqueModel._schema.drop_all()
        tables = self.database.get_tables()
        self.assertFalse(UniqueModel._meta.table_name in tables)

    def test_get_indexes(self):
        if False:
            i = 10
            return i + 15
        indexes = self.database.get_indexes('unique_model')
        data = [(index.name, index.columns, index.unique, index.table) for index in indexes if index.name not in ('unique_model_pkey', 'PRIMARY')]
        self.assertEqual(data, [('unique_model_name', ['name'], True, 'unique_model')])
        indexes = self.database.get_indexes('indexed_model')
        data = [(index.name, index.columns, index.unique, index.table) for index in indexes if index.name not in ('indexed_model_pkey', 'PRIMARY')]
        self.assertEqual(sorted(data), [('indexed_model_first_last', ['first', 'last'], False, 'indexed_model'), ('indexed_model_first_last_dob', ['first', 'last', 'dob'], True, 'indexed_model')])
        indexes = self.database.get_indexes('person')
        data = [(index.name, index.columns, index.unique) for index in indexes if index.name not in ('person_pkey', 'PRIMARY')]
        self.assertEqual(data, [('person_last_first', ['last', 'first'], False)])

    def test_get_columns(self):
        if False:
            return 10
        columns = self.database.get_columns('indexed_model')
        data = [(c.name, c.null, c.primary_key, c.table) for c in columns]
        self.assertEqual(data, [('id', False, True, 'indexed_model'), ('first', False, False, 'indexed_model'), ('last', False, False, 'indexed_model'), ('dob', False, False, 'indexed_model')])
        columns = self.database.get_columns('category')
        data = [(c.name, c.null, c.primary_key, c.table) for c in columns]
        self.assertEqual(data, [('name', False, True, 'category'), ('parent_id', True, False, 'category')])

    def test_get_primary_keys(self):
        if False:
            while True:
                i = 10
        primary_keys = self.database.get_primary_keys('users')
        self.assertEqual(primary_keys, ['id'])
        primary_keys = self.database.get_primary_keys('category')
        self.assertEqual(primary_keys, ['name'])

    @requires_models(Note)
    def test_get_views(self):
        if False:
            print('Hello World!')

        def normalize_view_meta(view_meta):
            if False:
                i = 10
                return i + 15
            sql_ws_norm = re.sub('[\\n\\s]+', ' ', view_meta.sql.strip('; '))
            return (view_meta.name, sql_ws_norm.replace('`peewee_test`.', '').replace('`notes`.', '').replace('notes.', '').replace('`', ''))

        def assertViews(expected):
            if False:
                while True:
                    i = 10
            self.database.execute_sql('CREATE VIEW notes_public AS SELECT content, ts FROM notes WHERE status = 1 ORDER BY ts DESC')
            self.database.execute_sql('CREATE VIEW notes_deleted AS SELECT content FROM notes WHERE status = 9 ORDER BY id DESC')
            try:
                views = self.database.get_views()
                normalized = sorted([normalize_view_meta(v) for v in views])
                self.assertEqual(normalized, expected)
                columns = self.database.get_columns('notes_deleted')
                self.assertEqual([c.name for c in columns], ['content'])
                columns = self.database.get_columns('notes_public')
                self.assertEqual([c.name for c in columns], ['content', 'ts'])
            finally:
                self.database.execute_sql('DROP VIEW notes_public;')
                self.database.execute_sql('DROP VIEW notes_deleted;')
        if IS_SQLITE:
            assertViews([('notes_deleted', 'CREATE VIEW notes_deleted AS SELECT content FROM notes WHERE status = 9 ORDER BY id DESC'), ('notes_public', 'CREATE VIEW notes_public AS SELECT content, ts FROM notes WHERE status = 1 ORDER BY ts DESC')])
        elif IS_MYSQL:
            assertViews([('notes_deleted', 'select content AS content from notes where status = 9 order by id desc'), ('notes_public', 'select content AS content,ts AS ts from notes where status = 1 order by ts desc')])
        elif IS_POSTGRESQL:
            assertViews([('notes_deleted', 'SELECT content FROM notes WHERE (status = 9) ORDER BY id DESC'), ('notes_public', 'SELECT content, ts FROM notes WHERE (status = 1) ORDER BY ts DESC')])
        elif IS_CRDB:
            assertViews([('notes_deleted', 'SELECT content FROM peewee_test.public.notes WHERE status = 9 ORDER BY id DESC'), ('notes_public', 'SELECT content, ts FROM peewee_test.public.notes WHERE status = 1 ORDER BY ts DESC')])

    @requires_models(User, Tweet, Category)
    def test_get_foreign_keys(self):
        if False:
            for i in range(10):
                print('nop')
        foreign_keys = self.database.get_foreign_keys('tweet')
        data = [(fk.column, fk.dest_table, fk.dest_column, fk.table) for fk in foreign_keys]
        self.assertEqual(data, [('user_id', 'users', 'id', 'tweet')])
        foreign_keys = self.database.get_foreign_keys('category')
        data = [(fk.column, fk.dest_table, fk.dest_column, fk.table) for fk in foreign_keys]
        self.assertEqual(data, [('parent_id', 'category', 'name', 'category')])

class TestSortModels(BaseTestCase):

    def test_sort_models(self):
        if False:
            i = 10
            return i + 15

        class A(Model):
            pass

        class B(Model):
            a = ForeignKeyField(A)

        class C(Model):
            b = ForeignKeyField(B)

        class D(Model):
            c = ForeignKeyField(C)

        class E(Model):
            pass
        models = [A, B, C, D, E]
        for list_of_models in permutations(models):
            sorted_models = sort_models(list_of_models)
            self.assertEqual(sorted_models, models)

class TestDBProxy(BaseTestCase):

    def test_proxy_context_manager(self):
        if False:
            print('Hello World!')
        db = Proxy()

        class User(Model):
            username = TextField()

            class Meta:
                database = db
        self.assertRaises(AttributeError, User.create_table)
        sqlite_db = SqliteDatabase(':memory:')
        db.initialize(sqlite_db)
        User.create_table()
        with db:
            self.assertFalse(db.is_closed())
        self.assertTrue(db.is_closed())

    def test_db_proxy(self):
        if False:
            print('Hello World!')
        db = Proxy()

        class BaseModel(Model):

            class Meta:
                database = db

        class User(BaseModel):
            username = TextField()

        class Tweet(BaseModel):
            user = ForeignKeyField(User, backref='tweets')
            message = TextField()
        sqlite_db = SqliteDatabase(':memory:')
        db.initialize(sqlite_db)
        self.assertEqual(User._meta.database.database, ':memory:')
        self.assertEqual(Tweet._meta.database.database, ':memory:')
        self.assertTrue(User._meta.database.is_closed())
        self.assertTrue(Tweet._meta.database.is_closed())
        sqlite_db.connect()
        self.assertFalse(User._meta.database.is_closed())
        self.assertFalse(Tweet._meta.database.is_closed())
        sqlite_db.close()

    def test_proxy_decorator(self):
        if False:
            while True:
                i = 10
        db = DatabaseProxy()

        @db.connection_context()
        def with_connection():
            if False:
                for i in range(10):
                    print('nop')
            self.assertFalse(db.is_closed())

        @db.atomic()
        def with_transaction():
            if False:
                return 10
            self.assertTrue(db.in_transaction())

        @db.manual_commit()
        def with_manual_commit():
            if False:
                return 10
            self.assertTrue(db.in_transaction())
        db.initialize(SqliteDatabase(':memory:'))
        with_connection()
        self.assertTrue(db.is_closed())
        with_transaction()
        self.assertFalse(db.in_transaction())
        with_manual_commit()
        self.assertFalse(db.in_transaction())

    def test_proxy_bind_ctx_callbacks(self):
        if False:
            print('Hello World!')
        db = Proxy()

        class BaseModel(Model):

            class Meta:
                database = db

        class Hook(BaseModel):
            data = BlobField()
        self.assertTrue(Hook.data._constructor is bytearray)

        class CustomSqliteDB(SqliteDatabase):
            sentinel = object()

            def get_binary_type(self):
                if False:
                    print('Hello World!')
                return self.sentinel
        custom_db = CustomSqliteDB(':memory:')
        with custom_db.bind_ctx([Hook]):
            self.assertTrue(Hook.data._constructor is custom_db.sentinel)
        self.assertTrue(Hook.data._constructor is bytearray)

class Data(TestModel):
    key = TextField()
    value = TextField()

    class Meta:
        schema = 'main'

class TestAttachDatabase(ModelTestCase):
    database = db_loader('sqlite3')
    requires = [Data]

    def test_attach(self):
        if False:
            while True:
                i = 10
        database = self.database
        Data.create(key='k1', value='v1')
        Data.create(key='k2', value='v2')
        database.attach(':memory:', 'cache')

        class CacheData(Data):

            class Meta:
                schema = 'cache'
        self.assertFalse(CacheData.table_exists())
        CacheData.create_table(safe=False)
        self.assertTrue(CacheData.table_exists())
        CacheData.insert_from(Data.select(), fields=[Data.id, Data.key, Data.value]).execute()
        query = Data.update({Data.value: Data.value + '-x'})
        self.assertEqual(query.execute(), 2)
        query = Data.select(Data.key, Data.value).order_by(Data.key)
        self.assertSQL(query, 'SELECT "t1"."key", "t1"."value" FROM "main"."data" AS "t1" ORDER BY "t1"."key"', [])
        self.assertEqual([v for (k, v) in query.tuples()], ['v1-x', 'v2-x'])
        query = CacheData.select(CacheData.key, CacheData.value).order_by(CacheData.key)
        self.assertSQL(query, 'SELECT "t1"."key", "t1"."value" FROM "cache"."cache_data" AS "t1" ORDER BY "t1"."key"', [])
        self.assertEqual([v for (k, v) in query.tuples()], ['v1', 'v2'])
        database.close()
        database.connect()
        self.assertFalse(CacheData.table_exists())
        curs = database.execute_sql('select * from cache.sqlite_master;')
        self.assertEqual(curs.fetchall(), [])
        CacheData.create_table(safe=False)
        self.assertEqual(CacheData.select().count(), 0)
        self.assertEqual(Data.select().count(), 2)

    def test_attach_detach(self):
        if False:
            for i in range(10):
                print('nop')
        database = self.database
        Data.create(key='k1', value='v1')
        Data.create(key='k2', value='v2')
        database.attach(':memory:', 'cache')
        curs = database.execute_sql('select * from cache.sqlite_master')
        self.assertEqual(curs.fetchall(), [])
        self.assertFalse(database.attach(':memory:', 'cache'))
        self.assertRaises(OperationalError, database.attach, 'foo.db', 'cache')
        self.assertTrue(database.detach('cache'))
        self.assertFalse(database.detach('cache'))
        self.assertRaises(OperationalError, database.execute_sql, 'select * from cache.sqlite_master')

    def test_sqlite_schema_support(self):
        if False:
            return 10

        class CacheData(Data):

            class Meta:
                schema = 'cache'
        self.database.attach(':memory:', 'cache')
        CacheData.create_table()
        tables = self.database.get_tables()
        self.assertEqual(tables, ['data'])
        tables = self.database.get_tables(schema='cache')
        self.assertEqual(tables, ['cache_data'])

class TestDatabaseConnection(DatabaseTestCase):

    def test_is_connection_usable(self):
        if False:
            return 10
        conn = self.database.connection()
        self.assertTrue(self.database.is_connection_usable())
        self.database.close()
        self.assertFalse(self.database.is_connection_usable())
        self.database.connect()
        self.assertTrue(self.database.is_connection_usable())

    @requires_postgresql
    def test_is_connection_usable_pg(self):
        if False:
            while True:
                i = 10
        self.database.execute_sql('drop table if exists foo')
        self.database.execute_sql('create table foo (data text not null)')
        self.assertTrue(self.database.is_connection_usable())
        with self.database.atomic() as txn:
            with self.assertRaises(IntegrityError):
                self.database.execute_sql('insert into foo (data) values (NULL)')
            self.assertFalse(self.database.is_closed())
            self.assertFalse(self.database.is_connection_usable())
            txn.rollback()
            self.assertTrue(self.database.is_connection_usable())
            curs = self.database.execute_sql('select * from foo')
            self.assertEqual(list(curs), [])
            self.database.execute_sql('drop table foo')

class TestExceptionWrapper(ModelTestCase):
    database = get_in_memory_db()
    requires = [User]

    def test_exception_wrapper(self):
        if False:
            return 10
        exc = None
        try:
            User.create(username=None)
        except IntegrityError as e:
            exc = e
        if exc is None:
            raise Exception('expected integrity error not raised')
        self.assertTrue(exc.orig.__module__ != 'peewee')

class TestModelPropertyHelper(BaseTestCase):

    def test_model_property(self):
        if False:
            i = 10
            return i + 15
        database = get_in_memory_db()

        class M1(database.Model):
            pass

        class M2(database.Model):
            pass

        class CM1(M1):
            pass
        for M in (M1, M2, CM1):
            self.assertTrue(M._meta.database is database)

    def test_model_property_on_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        db = DatabaseProxy()

        class M1(db.Model):
            pass

        class M2(db.Model):
            pass

        class CM1(M1):
            pass
        test_db = get_in_memory_db()
        db.initialize(test_db)
        for M in (M1, M2, CM1):
            self.assertEqual(M._meta.database.database, ':memory:')
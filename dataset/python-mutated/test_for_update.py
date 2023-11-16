"""Test MySQL FOR UPDATE behavior.

See #4246

"""
import contextlib
from sqlalchemy import Column
from sqlalchemy import exc
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import literal_column
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy import update
from sqlalchemy.dialects.mysql import base as mysql
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.sql import column
from sqlalchemy.sql import table
from sqlalchemy.testing import AssertsCompiledSQL
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures

class MySQLForUpdateLockingTest(fixtures.DeclarativeMappedTest):
    __backend__ = True
    __only_on__ = ('mysql', 'mariadb')
    __requires__ = ('mysql_for_update',)

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')
        Base = cls.DeclarativeBasic

        class A(Base):
            __tablename__ = 'a'
            id = Column(Integer, primary_key=True)
            x = Column(Integer)
            y = Column(Integer)
            bs = relationship('B')
            __table_args__ = {'mysql_engine': 'InnoDB', 'mariadb_engine': 'InnoDB'}

        class B(Base):
            __tablename__ = 'b'
            id = Column(Integer, primary_key=True)
            a_id = Column(ForeignKey('a.id'))
            x = Column(Integer)
            y = Column(Integer)
            __table_args__ = {'mysql_engine': 'InnoDB', 'mariadb_engine': 'InnoDB'}

    @classmethod
    def insert_data(cls, connection):
        if False:
            while True:
                i = 10
        A = cls.classes.A
        B = cls.classes.B
        s = Session(connection)
        s.add_all([A(x=5, y=5, bs=[B(x=4, y=4), B(x=2, y=8), B(x=7, y=1)]), A(x=7, y=5, bs=[B(x=4, y=4), B(x=5, y=8)])])
        s.commit()

    @contextlib.contextmanager
    def run_test(self):
        if False:
            print('Hello World!')
        connection = testing.db.connect()
        connection.exec_driver_sql('set innodb_lock_wait_timeout=1')
        try:
            yield Session(bind=connection)
        finally:
            connection.rollback()
            connection.close()

    def _assert_a_is_locked(self, should_be_locked):
        if False:
            while True:
                i = 10
        A = self.classes.A
        with testing.db.begin() as alt_trans:
            alt_trans.exec_driver_sql('set innodb_lock_wait_timeout=1')
            try:
                alt_trans.execute(update(A).values(x=15, y=19))
            except (exc.InternalError, exc.OperationalError) as err:
                assert 'Lock wait timeout exceeded' in str(err)
                assert should_be_locked
            else:
                assert not should_be_locked

    def _assert_b_is_locked(self, should_be_locked):
        if False:
            print('Hello World!')
        B = self.classes.B
        with testing.db.begin() as alt_trans:
            alt_trans.exec_driver_sql('set innodb_lock_wait_timeout=1')
            try:
                alt_trans.execute(update(B).values(x=15, y=19))
            except (exc.InternalError, exc.OperationalError) as err:
                assert 'Lock wait timeout exceeded' in str(err)
                assert should_be_locked
            else:
                assert not should_be_locked

    def test_basic_lock(self):
        if False:
            print('Hello World!')
        A = self.classes.A
        with self.run_test() as s:
            s.query(A).with_for_update().all()
            self._assert_a_is_locked(True)

    def test_basic_not_lock(self):
        if False:
            for i in range(10):
                print('nop')
        A = self.classes.A
        with self.run_test() as s:
            s.query(A).all()
            self._assert_a_is_locked(False)

    def test_joined_lock_subquery(self):
        if False:
            while True:
                i = 10
        A = self.classes.A
        with self.run_test() as s:
            s.query(A).options(joinedload(A.bs)).with_for_update().first()
            self._assert_a_is_locked(True)
            self._assert_b_is_locked(True)

    def test_joined_lock_subquery_inner_for_update(self):
        if False:
            for i in range(10):
                print('nop')
        A = self.classes.A
        B = self.classes.B
        with self.run_test() as s:
            q = s.query(A).with_for_update().subquery()
            s.query(q).join(B).all()
            self._assert_a_is_locked(True)
            self._assert_b_is_locked(False)

    def test_joined_lock_subquery_inner_for_update_outer(self):
        if False:
            return 10
        A = self.classes.A
        B = self.classes.B
        with self.run_test() as s:
            q = s.query(A).with_for_update().subquery()
            s.query(q).join(B).with_for_update().all()
            self._assert_a_is_locked(True)
            self._assert_b_is_locked(True)

    def test_joined_lock_subquery_order_for_update_outer(self):
        if False:
            i = 10
            return i + 15
        A = self.classes.A
        B = self.classes.B
        with self.run_test() as s:
            q = s.query(A).order_by(A.id).subquery()
            s.query(q).join(B).with_for_update().all()
            self._assert_a_is_locked(False)
            self._assert_b_is_locked(True)

    def test_joined_lock_no_subquery(self):
        if False:
            print('Hello World!')
        A = self.classes.A
        with self.run_test() as s:
            s.query(A).options(joinedload(A.bs)).with_for_update().all()
            self._assert_a_is_locked(True)
            self._assert_b_is_locked(True)

class MySQLForUpdateCompileTest(fixtures.TestBase, AssertsCompiledSQL):
    __dialect__ = mysql.dialect()
    table1 = table('mytable', column('myid'), column('name'), column('description'))
    table2 = table('table2', column('mytable_id'))
    join = table2.join(table1, table2.c.mytable_id == table1.c.myid)
    for_update_of_dialect = mysql.dialect()
    for_update_of_dialect.server_version_info = (8, 0, 0)
    for_update_of_dialect.supports_for_update_of = True

    def test_for_update_basic(self):
        if False:
            return 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE')

    def test_for_update_read(self):
        if False:
            return 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE')

    def test_for_update_skip_locked(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(skip_locked=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE SKIP LOCKED')

    def test_for_update_read_and_skip_locked(self):
        if False:
            i = 10
            return i + 15
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, skip_locked=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE SKIP LOCKED')

    def test_for_update_nowait(self):
        if False:
            while True:
                i = 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(nowait=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE NOWAIT')

    def test_for_update_read_and_nowait(self):
        if False:
            return 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, nowait=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE NOWAIT')

    def test_for_update_of_nowait(self):
        if False:
            return 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(of=self.table1, nowait=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE OF mytable NOWAIT', dialect=self.for_update_of_dialect)

    def test_for_update_of_basic(self):
        if False:
            print('Hello World!')
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(of=self.table1), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE OF mytable', dialect=self.for_update_of_dialect)

    def test_for_update_of_skip_locked(self):
        if False:
            print('Hello World!')
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(of=self.table1, skip_locked=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE OF mytable SKIP LOCKED', dialect=self.for_update_of_dialect)

    def test_for_update_of_join_one(self):
        if False:
            i = 10
            return i + 15
        self.assert_compile(self.join.select().where(self.table2.c.mytable_id == 7).with_for_update(of=[self.join]), 'SELECT table2.mytable_id, mytable.myid, mytable.name, mytable.description FROM table2 INNER JOIN mytable ON table2.mytable_id = mytable.myid WHERE table2.mytable_id = %s FOR UPDATE OF mytable, table2', dialect=self.for_update_of_dialect)

    def test_for_update_of_column_list_aliased(self):
        if False:
            i = 10
            return i + 15
        ta = self.table1.alias()
        self.assert_compile(ta.select().where(ta.c.myid == 7).with_for_update(of=[ta.c.myid, ta.c.name]), 'SELECT mytable_1.myid, mytable_1.name, mytable_1.description FROM mytable AS mytable_1 WHERE mytable_1.myid = %s FOR UPDATE OF mytable_1', dialect=self.for_update_of_dialect)

    def test_for_update_of_join_aliased(self):
        if False:
            i = 10
            return i + 15
        ta = self.table1.alias()
        alias_join = self.table2.join(ta, self.table2.c.mytable_id == ta.c.myid)
        self.assert_compile(alias_join.select().where(self.table2.c.mytable_id == 7).with_for_update(of=[alias_join]), 'SELECT table2.mytable_id, mytable_1.myid, mytable_1.name, mytable_1.description FROM table2 INNER JOIN mytable AS mytable_1 ON table2.mytable_id = mytable_1.myid WHERE table2.mytable_id = %s FOR UPDATE OF mytable_1, table2', dialect=self.for_update_of_dialect)

    def test_for_update_of_read_nowait(self):
        if False:
            print('Hello World!')
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, of=self.table1, nowait=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE OF mytable NOWAIT', dialect=self.for_update_of_dialect)

    def test_for_update_of_read_skip_locked(self):
        if False:
            return 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, of=self.table1, skip_locked=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE OF mytable SKIP LOCKED', dialect=self.for_update_of_dialect)

    def test_for_update_of_read_nowait_column_list(self):
        if False:
            i = 10
            return i + 15
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, of=[self.table1.c.myid, self.table1.c.name], nowait=True), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE OF mytable NOWAIT', dialect=self.for_update_of_dialect)

    def test_for_update_of_read(self):
        if False:
            print('Hello World!')
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(read=True, of=self.table1), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s LOCK IN SHARE MODE OF mytable', dialect=self.for_update_of_dialect)

    def test_for_update_textual_of(self):
        if False:
            while True:
                i = 10
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(of=text('mytable')), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE OF mytable', dialect=self.for_update_of_dialect)
        self.assert_compile(self.table1.select().where(self.table1.c.myid == 7).with_for_update(of=literal_column('mytable')), 'SELECT mytable.myid, mytable.name, mytable.description FROM mytable WHERE mytable.myid = %s FOR UPDATE OF mytable', dialect=self.for_update_of_dialect)

class SkipLockedTest(fixtures.TablesTest):
    __only_on__ = ('mysql', 'mariadb')
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            i = 10
            return i + 15
        Table('stuff', metadata, Column('id', Integer, primary_key=True), Column('value', Integer))

    @testing.only_on(['mysql>=8', 'mariadb>=10.6'])
    def test_skip_locked(self, connection):
        if False:
            i = 10
            return i + 15
        stuff = self.tables.stuff
        stmt = stuff.select().with_for_update(skip_locked=True)
        connection.execute(stmt).fetchall()

    @testing.only_on(['mysql<8', 'mariadb<10.6'])
    def test_unsupported_skip_locked(self, connection):
        if False:
            i = 10
            return i + 15
        stuff = self.tables.stuff
        stmt = stuff.select().with_for_update(skip_locked=True)
        with expect_raises_message(ProgrammingError, 'You have an error in your SQL syntax'):
            connection.execute(stmt).fetchall()
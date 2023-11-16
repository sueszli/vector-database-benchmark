from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import exc
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import insert_sentinel
from sqlalchemy import Integer
from sqlalchemy import join
from sqlalchemy import select
from sqlalchemy import testing
from sqlalchemy.orm import aliased
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import noload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_warnings
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.assertions import expect_raises_message
from sqlalchemy.testing.assertsql import CompiledSQL
from sqlalchemy.testing.entities import ComparableEntity
from sqlalchemy.testing.fixtures import fixture_session

class PartitionByFixture(fixtures.DeclarativeMappedTest):

    @classmethod
    def setup_classes(cls):
        if False:
            print('Hello World!')
        Base = cls.DeclarativeBasic

        class A(Base):
            __tablename__ = 'a'
            id = Column(Integer, primary_key=True)

        class B(Base):
            __tablename__ = 'b'
            id = Column(Integer, primary_key=True)
            a_id = Column(ForeignKey('a.id'))
            cs = relationship('C')

        class C(Base):
            __tablename__ = 'c'
            id = Column(Integer, primary_key=True)
            b_id = Column(ForeignKey('b.id'))
            _sentinel = insert_sentinel()
        partition = select(B, func.row_number().over(order_by=B.id, partition_by=B.a_id).label('index')).alias()
        cls.partitioned_b = partitioned_b = aliased(B, alias=partition)
        A.partitioned_bs = relationship(partitioned_b, primaryjoin=and_(partitioned_b.a_id == A.id, partition.c.index < 10))

    @classmethod
    def insert_data(cls, connection):
        if False:
            while True:
                i = 10
        (A, B, C) = cls.classes('A', 'B', 'C')
        s = Session(connection)
        s.add_all([A(id=i) for i in range(1, 4)])
        s.flush()
        s.add_all([B(a_id=i, cs=[C(), C()]) for i in range(1, 4) for j in range(1, 21)])
        s.commit()

class AliasedClassRelationshipTest(PartitionByFixture, testing.AssertsCompiledSQL):
    __requires__ = ('window_functions',)
    __dialect__ = 'default'

    def test_lazyload(self):
        if False:
            print('Hello World!')
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)

        def go():
            if False:
                while True:
                    i = 10
            for a1 in s.query(A):
                eq_(len(a1.partitioned_bs), 9)
                for b in a1.partitioned_bs:
                    eq_(len(b.cs), 2)
        self.assert_sql_count(testing.db, go, 31)

    def test_join_one(self):
        if False:
            print('Hello World!')
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)
        q = s.query(A).join(A.partitioned_bs)
        self.assert_compile(q, 'SELECT a.id AS a_id FROM a JOIN (SELECT b.id AS id, b.a_id AS a_id, row_number() OVER (PARTITION BY b.a_id ORDER BY b.id) AS index FROM b) AS anon_1 ON anon_1.a_id = a.id AND anon_1.index < :index_1')

    def test_join_two(self):
        if False:
            return 10
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)
        q = s.query(A, A.partitioned_bs.entity).join(A.partitioned_bs)
        self.assert_compile(q, 'SELECT a.id AS a_id, anon_1.id AS anon_1_id, anon_1.a_id AS anon_1_a_id FROM a JOIN (SELECT b.id AS id, b.a_id AS a_id, row_number() OVER (PARTITION BY b.a_id ORDER BY b.id) AS index FROM b) AS anon_1 ON anon_1.a_id = a.id AND anon_1.index < :index_1')

    def test_selectinload_w_noload_after(self):
        if False:
            return 10
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)

        def go():
            if False:
                while True:
                    i = 10
            for a1 in s.query(A).options(noload('*'), selectinload(A.partitioned_bs)):
                for b in a1.partitioned_bs:
                    eq_(b.cs, [])
        self.assert_sql_count(testing.db, go, 2)

    @testing.combinations('ac_attribute', 'ac_attr_w_of_type')
    def test_selectinload_w_joinedload_after(self, calling_style):
        if False:
            for i in range(10):
                print('nop')
        'test has been enhanced to also test #7224'
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)
        partitioned_b = self.partitioned_b
        if calling_style == 'ac_attribute':
            opt = selectinload(A.partitioned_bs).joinedload(partitioned_b.cs)
        elif calling_style == 'ac_attr_w_of_type':
            opt = selectinload(A.partitioned_bs.of_type(partitioned_b)).joinedload(partitioned_b.cs)
        else:
            assert False

        def go():
            if False:
                print('Hello World!')
            for a1 in s.query(A).options(opt):
                for b in a1.partitioned_bs:
                    eq_(len(b.cs), 2)
        self.assert_sql_count(testing.db, go, 2)

    @testing.combinations(True, False)
    def test_selectinload_w_joinedload_after_base_target_fails(self, use_of_type):
        if False:
            for i in range(10):
                print('nop')
        (A, B, C) = self.classes('A', 'B', 'C')
        s = Session(testing.db)
        partitioned_b = self.partitioned_b
        with expect_raises_message(exc.ArgumentError, 'ORM mapped entity or attribute "B.cs" does not link from relationship "A.partitioned_bs.of_type\\(aliased\\(B\\)\\)"'):
            if use_of_type:
                opt = selectinload(A.partitioned_bs.of_type(partitioned_b)).joinedload(B.cs)
            else:
                opt = selectinload(A.partitioned_bs).joinedload(B.cs)
            q = s.query(A).options(opt)
            q._compile_context()

class AltSelectableTest(fixtures.DeclarativeMappedTest, testing.AssertsCompiledSQL):
    __dialect__ = 'default'

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')
        Base = cls.DeclarativeBasic

        class A(ComparableEntity, Base):
            __tablename__ = 'a'
            id = Column(Integer, primary_key=True)
            b_id = Column(ForeignKey('b.id'))

        class B(ComparableEntity, Base):
            __tablename__ = 'b'
            id = Column(Integer, primary_key=True)

        class C(ComparableEntity, Base):
            __tablename__ = 'c'
            id = Column(Integer, primary_key=True)
            a_id = Column(ForeignKey('a.id'))

        class D(ComparableEntity, Base):
            __tablename__ = 'd'
            id = Column(Integer, primary_key=True)
            c_id = Column(ForeignKey('c.id'))
            b_id = Column(ForeignKey('b.id'))
        j = join(B, D, D.b_id == B.id).join(C, C.id == D.c_id)
        B_viacd = aliased(B, j, flat=True)
        A.b = relationship(B_viacd, primaryjoin=A.b_id == j.c.b_id)

    @classmethod
    def insert_data(cls, connection):
        if False:
            for i in range(10):
                print('nop')
        (A, B, C, D) = cls.classes('A', 'B', 'C', 'D')
        sess = Session(connection)
        for obj in [B(id=1), A(id=1, b_id=1), C(id=1, a_id=1), D(id=1, c_id=1, b_id=1)]:
            sess.add(obj)
            sess.flush()
        sess.commit()

    def test_lazyload(self):
        if False:
            i = 10
            return i + 15
        (A, B) = self.classes('A', 'B')
        sess = fixture_session()
        a1 = sess.query(A).first()
        with self.sql_execution_asserter() as asserter:
            eq_(a1.b, B(id=1))
        asserter.assert_(CompiledSQL('SELECT b.id AS b_id FROM b JOIN d ON d.b_id = b.id JOIN c ON c.id = d.c_id WHERE :param_1 = b.id', [{'param_1': 1}]))

    def test_joinedload(self):
        if False:
            print('Hello World!')
        (A, B) = self.classes('A', 'B')
        sess = fixture_session()
        with self.sql_execution_asserter() as asserter:
            a1 = sess.query(A).options(joinedload(A.b)).first()
            eq_(a1.b, B(id=1))
        asserter.assert_(CompiledSQL('SELECT a.id AS a_id, a.b_id AS a_b_id, b_1.id AS b_1_id FROM a LEFT OUTER JOIN (b AS b_1 JOIN d AS d_1 ON d_1.b_id = b_1.id JOIN c AS c_1 ON c_1.id = d_1.c_id) ON a.b_id = b_1.id LIMIT :param_1', [{'param_1': 1}]))

    def test_selectinload(self):
        if False:
            i = 10
            return i + 15
        (A, B) = self.classes('A', 'B')
        sess = fixture_session()
        with self.sql_execution_asserter() as asserter:
            a1 = sess.query(A).options(selectinload(A.b)).first()
            eq_(a1.b, B(id=1))
        asserter.assert_(CompiledSQL('SELECT a.id AS a_id, a.b_id AS a_b_id FROM a LIMIT :param_1', [{'param_1': 1}]), CompiledSQL('SELECT a_1.id AS a_1_id, b.id AS b_id FROM a AS a_1 JOIN (b JOIN d ON d.b_id = b.id JOIN c ON c.id = d.c_id) ON a_1.b_id = b.id WHERE a_1.id IN (__[POSTCOMPILE_primary_keys])', [{'primary_keys': [1]}]))

    def test_join(self):
        if False:
            print('Hello World!')
        (A, B) = self.classes('A', 'B')
        sess = fixture_session()
        self.assert_compile(sess.query(A).join(A.b), 'SELECT a.id AS a_id, a.b_id AS a_b_id FROM a JOIN (b JOIN d ON d.b_id = b.id JOIN c ON c.id = d.c_id) ON a.b_id = b.id')

class StructuralEagerLoadCycleTest(fixtures.DeclarativeMappedTest):

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')
        Base = cls.DeclarativeBasic

        class A(Base):
            __tablename__ = 'a'
            id = Column(Integer, primary_key=True)
            bs = relationship(lambda : B, back_populates='a')

        class B(Base):
            __tablename__ = 'b'
            id = Column(Integer, primary_key=True)
            a_id = Column(ForeignKey('a.id'))
            a = relationship(A, lazy='joined', back_populates='bs')
        partitioned_b = aliased(B)
        A.partitioned_bs = relationship(partitioned_b, lazy='selectin', viewonly=True)

    @classmethod
    def insert_data(cls, connection):
        if False:
            for i in range(10):
                print('nop')
        (A, B) = cls.classes('A', 'B')
        s = Session(connection)
        a = A()
        a.bs = [B() for _ in range(5)]
        s.add(a)
        s.commit()

    @testing.variation('ensure_no_warning', [True, False])
    def test_no_endless_loop(self, ensure_no_warning):
        if False:
            while True:
                i = 10
        'test #9590'
        A = self.classes.A
        sess = fixture_session()
        results = sess.scalars(select(A))
        if ensure_no_warning:
            a = results.first()
        else:
            with expect_warnings('Loader depth for query is excessively deep', assert_=False):
                a = results.first()
        a.bs
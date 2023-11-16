from sqlalchemy import desc
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import testing
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table

class UnicodeSchemaTest(fixtures.TablesTest):
    __requires__ = ('unicode_ddl',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            while True:
                i = 10
        global t1, t2, t3
        t1 = Table('unitable1', metadata, Column('méil', Integer, primary_key=True), Column('測試', Integer), test_needs_fk=True)
        t2 = Table('Unitéble2', metadata, Column('méil', Integer, primary_key=True, key='a'), Column('測試', Integer, ForeignKey('unitable1.méil'), key='b'), test_needs_fk=True)
        if testing.against('sqlite'):
            t3 = Table('測試', metadata, Column('測試_id', Integer, primary_key=True, autoincrement=False), Column('unitable1_測試', Integer, ForeignKey('unitable1.測試')), Column('Unitéble2_b', Integer, ForeignKey('Unitéble2.b')), Column('測試_self', Integer, ForeignKey('測試.測試_id')), test_needs_fk=True)
        else:
            t3 = Table('測試', metadata, Column('測試_id', Integer, primary_key=True, autoincrement=False), Column('unitable1_測試', Integer), Column('Unitéble2_b', Integer), Column('測試_self', Integer), test_needs_fk=True)

    def test_insert(self, connection):
        if False:
            return 10
        connection.execute(t1.insert(), {'méil': 1, '測試': 5})
        connection.execute(t2.insert(), {'a': 1, 'b': 1})
        connection.execute(t3.insert(), {'測試_id': 1, 'unitable1_測試': 5, 'Unitéble2_b': 1, '測試_self': 1})
        eq_(connection.execute(t1.select()).fetchall(), [(1, 5)])
        eq_(connection.execute(t2.select()).fetchall(), [(1, 1)])
        eq_(connection.execute(t3.select()).fetchall(), [(1, 5, 1, 1)])

    def test_col_targeting(self, connection):
        if False:
            for i in range(10):
                print('nop')
        connection.execute(t1.insert(), {'méil': 1, '測試': 5})
        connection.execute(t2.insert(), {'a': 1, 'b': 1})
        connection.execute(t3.insert(), {'測試_id': 1, 'unitable1_測試': 5, 'Unitéble2_b': 1, '測試_self': 1})
        row = connection.execute(t1.select()).first()
        eq_(row._mapping[t1.c['méil']], 1)
        eq_(row._mapping[t1.c['測試']], 5)
        row = connection.execute(t2.select()).first()
        eq_(row._mapping[t2.c['a']], 1)
        eq_(row._mapping[t2.c['b']], 1)
        row = connection.execute(t3.select()).first()
        eq_(row._mapping[t3.c['測試_id']], 1)
        eq_(row._mapping[t3.c['unitable1_測試']], 5)
        eq_(row._mapping[t3.c['Unitéble2_b']], 1)
        eq_(row._mapping[t3.c['測試_self']], 1)

    def test_reflect(self, connection):
        if False:
            for i in range(10):
                print('nop')
        connection.execute(t1.insert(), {'méil': 2, '測試': 7})
        connection.execute(t2.insert(), {'a': 2, 'b': 2})
        connection.execute(t3.insert(), {'測試_id': 2, 'unitable1_測試': 7, 'Unitéble2_b': 2, '測試_self': 2})
        meta = MetaData()
        tt1 = Table(t1.name, meta, autoload_with=connection)
        tt2 = Table(t2.name, meta, autoload_with=connection)
        tt3 = Table(t3.name, meta, autoload_with=connection)
        connection.execute(tt1.insert(), {'méil': 1, '測試': 5})
        connection.execute(tt2.insert(), {'méil': 1, '測試': 1})
        connection.execute(tt3.insert(), {'測試_id': 1, 'unitable1_測試': 5, 'Unitéble2_b': 1, '測試_self': 1})
        eq_(connection.execute(tt1.select().order_by(desc('méil'))).fetchall(), [(2, 7), (1, 5)])
        eq_(connection.execute(tt2.select().order_by(desc('méil'))).fetchall(), [(2, 2), (1, 1)])
        eq_(connection.execute(tt3.select().order_by(desc('測試_id'))).fetchall(), [(2, 7, 2, 2), (1, 5, 1, 1)])

    def test_repr(self):
        if False:
            while True:
                i = 10
        meta = MetaData()
        t = Table('測試', meta, Column('測試_id', Integer))
        eq_(repr(t), "Table('測試', MetaData(), Column('測試_id', Integer(), table=<測試>), schema=None)")
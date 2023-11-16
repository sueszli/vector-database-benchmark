from __future__ import annotations
from sqlalchemy import bindparam
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import Identity
from sqlalchemy import insert
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from . import Profiler
'This series of tests illustrates different ways to INSERT a large number\nof rows in bulk.\n\n\n'
Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Integer, Identity(), primary_key=True)
    name = Column(String(255))
    description = Column(String(255))
Profiler.init('bulk_inserts', num=100000)

@Profiler.setup
def setup_database(dburl, echo, num):
    if False:
        for i in range(10):
            print('nop')
    global engine
    engine = create_engine(dburl, echo=echo)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

@Profiler.profile
def test_flush_no_pk(n):
    if False:
        return 10
    'INSERT statements via the ORM (batched with RETURNING if available),\n    fetching generated row id'
    session = Session(bind=engine)
    for chunk in range(0, n, 1000):
        session.add_all([Customer(name='customer name %d' % i, description='customer description %d' % i) for i in range(chunk, chunk + 1000)])
        session.flush()
    session.commit()

@Profiler.profile
def test_flush_pk_given(n):
    if False:
        for i in range(10):
            print('nop')
    'Batched INSERT statements via the ORM, PKs already defined'
    session = Session(bind=engine)
    for chunk in range(0, n, 1000):
        session.add_all([Customer(id=i + 1, name='customer name %d' % i, description='customer description %d' % i) for i in range(chunk, chunk + 1000)])
        session.flush()
    session.commit()

@Profiler.profile
def test_orm_bulk_insert(n):
    if False:
        i = 10
        return i + 15
    'Batched INSERT statements via the ORM in "bulk", not returning rows'
    session = Session(bind=engine)
    session.execute(insert(Customer), [{'name': 'customer name %d' % i, 'description': 'customer description %d' % i} for i in range(n)])
    session.commit()

@Profiler.profile
def test_orm_insert_returning(n):
    if False:
        print('Hello World!')
    'Batched INSERT statements via the ORM in "bulk", returning new Customer\n    objects'
    session = Session(bind=engine)
    customer_result = session.scalars(insert(Customer).returning(Customer), [{'name': 'customer name %d' % i, 'description': 'customer description %d' % i} for i in range(n)])
    customers = customer_result.all()
    session.commit()

@Profiler.profile
def test_core_insert(n):
    if False:
        print('Hello World!')
    'A single Core INSERT construct inserting mappings in bulk.'
    with engine.begin() as conn:
        conn.execute(Customer.__table__.insert(), [dict(name='customer name %d' % i, description='customer description %d' % i) for i in range(n)])

@Profiler.profile
def test_dbapi_raw(n):
    if False:
        return 10
    "The DBAPI's API inserting rows in bulk."
    conn = engine.pool._creator()
    cursor = conn.cursor()
    compiled = Customer.__table__.insert().values(name=bindparam('name'), description=bindparam('description')).compile(dialect=engine.dialect)
    if compiled.positional:
        args = (('customer name %d' % i, 'customer description %d' % i) for i in range(n))
    else:
        args = (dict(name='customer name %d' % i, description='customer description %d' % i) for i in range(n))
    cursor.executemany(str(compiled), list(args))
    conn.commit()
    conn.close()
if __name__ == '__main__':
    Profiler.main()
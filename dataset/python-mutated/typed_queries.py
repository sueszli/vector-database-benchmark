from __future__ import annotations
from typing import Tuple
from sqlalchemy import Column
from sqlalchemy import column
from sqlalchemy import create_engine
from sqlalchemy import delete
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import insert
from sqlalchemy import Integer
from sqlalchemy import join
from sqlalchemy import MetaData
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import text
from sqlalchemy import update
from sqlalchemy.orm import aliased
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'user'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    data: Mapped[str]
user_table = Table('user', MetaData(), Column('id', Integer, primary_key=True), Column('name', String, primary_key=True))
session = Session()
e = create_engine('sqlite://')
connection = e.connect()

def t_select_1() -> None:
    if False:
        for i in range(10):
            print('nop')
    stmt = select(User.id, User.name).filter(User.id == 5)
    reveal_type(stmt)
    result = session.execute(stmt)
    reveal_type(result)

def t_select_2() -> None:
    if False:
        while True:
            i = 10
    stmt = select(User).filter(User.id == 5).limit(1).offset(3).offset(None).limit(None).limit(User.id).offset(User.id).fetch(1).fetch(None).fetch(User.id)
    reveal_type(stmt)
    result = session.execute(stmt)
    reveal_type(result)

def t_select_3() -> None:
    if False:
        print('Hello World!')
    ua = aliased(User)
    ua(id=1, name='foo')
    reveal_type(ua)
    stmt = select(ua.id, ua.name).filter(User.id == 5)
    reveal_type(stmt)
    result = session.execute(stmt)
    reveal_type(result)

def t_select_4() -> None:
    if False:
        while True:
            i = 10
    ua = aliased(User)
    stmt = select(ua, User).filter(User.id == 5)
    reveal_type(stmt)
    result = session.execute(stmt)
    reveal_type(result)

def t_legacy_query_single_entity() -> None:
    if False:
        i = 10
        return i + 15
    q1 = session.query(User).filter(User.id == 5)
    reveal_type(q1)
    reveal_type(q1.one())
    reveal_type(q1.all())
    reveal_type(q1.only_return_tuples(True).all())
    reveal_type(q1.tuples().all())

def t_legacy_query_cols_1() -> None:
    if False:
        print('Hello World!')
    q1 = session.query(User.id, User.name).filter(User.id == 5)
    reveal_type(q1)
    reveal_type(q1.one())
    r1 = q1.one()
    (x, y) = r1.t
    reveal_type(x)
    reveal_type(y)

def t_legacy_query_cols_tupleq_1() -> None:
    if False:
        return 10
    q1 = session.query(User.id, User.name).filter(User.id == 5)
    reveal_type(q1)
    q2 = q1.tuples()
    reveal_type(q2.one())
    r1 = q2.one()
    (x, y) = r1
    reveal_type(x)
    reveal_type(y)

def t_legacy_query_cols_1_with_entities() -> None:
    if False:
        while True:
            i = 10
    q1 = session.query(User).filter(User.id == 5)
    reveal_type(q1)
    q2 = q1.with_entities(User.id, User.name)
    reveal_type(q2)
    reveal_type(q2.one())
    r1 = q2.one()
    (x, y) = r1.t
    reveal_type(x)
    reveal_type(y)

def t_select_with_only_cols() -> None:
    if False:
        for i in range(10):
            print('nop')
    q1 = select(User).where(User.id == 5)
    reveal_type(q1)
    q2 = q1.with_only_columns(User.id, User.name)
    reveal_type(q2)
    row = connection.execute(q2).one()
    reveal_type(row)
    (x, y) = row.t
    reveal_type(x)
    reveal_type(y)

def t_legacy_query_cols_2() -> None:
    if False:
        while True:
            i = 10
    a1 = aliased(User)
    q1 = session.query(User, a1, User.name).filter(User.id == 5)
    reveal_type(q1)
    reveal_type(q1.one())
    r1 = q1.one()
    (x, y, z) = r1.t
    reveal_type(x)
    reveal_type(y)
    reveal_type(z)

def t_legacy_query_cols_2_with_entities() -> None:
    if False:
        while True:
            i = 10
    q1 = session.query(User)
    reveal_type(q1)
    a1 = aliased(User)
    q2 = q1.with_entities(User, a1, User.name).filter(User.id == 5)
    reveal_type(q2)
    reveal_type(q2.one())
    r1 = q2.one()
    (x, y, z) = r1.t
    reveal_type(x)
    reveal_type(y)
    reveal_type(z)

def t_select_add_col_loses_type() -> None:
    if False:
        while True:
            i = 10
    q1 = select(User.id, User.name).filter(User.id == 5)
    q2 = q1.add_columns(User.data)
    reveal_type(q2)

def t_legacy_query_add_col_loses_type() -> None:
    if False:
        return 10
    q1 = session.query(User.id, User.name).filter(User.id == 5)
    q2 = q1.add_columns(User.data)
    reveal_type(q2)
    ua = aliased(User)
    q3 = q1.add_entity(ua)
    reveal_type(q3)

def t_legacy_query_scalar_subquery() -> None:
    if False:
        print('Hello World!')
    'scalar subquery should receive the type if first element is a\n    column only'
    q1 = session.query(User.id)
    q2 = q1.scalar_subquery()
    reveal_type(q2)
    q3 = session.query(User)
    q4 = q3.scalar_subquery()
    reveal_type(q4)
    q5 = session.query(User, User.name)
    q6 = q5.scalar_subquery()
    reveal_type(q6)
    q7 = session.query(User).only_return_tuples(True)
    q8 = q7.scalar_subquery()
    reveal_type(q8)

def t_select_scalar_subquery() -> None:
    if False:
        print('Hello World!')
    'scalar subquery should receive the type if first element is a\n    column only'
    s1 = select(User.id)
    s2 = s1.scalar_subquery()
    reveal_type(s2)
    s3 = select(User)
    s4 = s3.scalar_subquery()
    reveal_type(s4)

def t_select_w_core_selectables() -> None:
    if False:
        for i in range(10):
            print('nop')
    'things that come from .c. or are FromClause objects currently are not\n    typed.  Make sure we are still getting Select at least.\n\n    '
    s1 = select(User.id, User.name).subquery()
    reveal_type(s1.c.name)
    s2 = select(User.id, s1.c.name)
    reveal_type(s2)
    s2_typed: Select[Tuple[int, str]] = select(User.id, s1.c.name)
    reveal_type(s2_typed)
    s3 = select(s1)
    reveal_type(s3)
    t1 = User.__table__
    assert t1 is not None
    reveal_type(t1)
    s4 = select(t1)
    reveal_type(s4)

def t_dml_insert() -> None:
    if False:
        i = 10
        return i + 15
    s1 = insert(User).returning(User.id, User.name)
    r1 = session.execute(s1)
    reveal_type(r1)
    s2 = insert(User).returning(User)
    r2 = session.execute(s2)
    reveal_type(r2)
    s3 = insert(User).returning(func.foo(), column('q'))
    reveal_type(s3)
    r3 = session.execute(s3)
    reveal_type(r3)

def t_dml_bare_insert() -> None:
    if False:
        return 10
    s1 = insert(User)
    r1 = session.execute(s1)
    reveal_type(r1)
    reveal_type(r1.rowcount)

def t_dml_bare_update() -> None:
    if False:
        print('Hello World!')
    s1 = update(User)
    r1 = session.execute(s1)
    reveal_type(r1)
    reveal_type(r1.rowcount)

def t_dml_update_with_values() -> None:
    if False:
        print('Hello World!')
    s1 = update(User).values({User.id: 123, User.data: 'value'})
    r1 = session.execute(s1)
    reveal_type(r1)
    reveal_type(r1.rowcount)

def t_dml_bare_delete() -> None:
    if False:
        return 10
    s1 = delete(User)
    r1 = session.execute(s1)
    reveal_type(r1)
    reveal_type(r1.rowcount)

def t_dml_update() -> None:
    if False:
        return 10
    s1 = update(User).returning(User.id, User.name)
    r1 = session.execute(s1)
    reveal_type(r1)

def t_dml_delete() -> None:
    if False:
        print('Hello World!')
    s1 = delete(User).returning(User.id, User.name)
    r1 = session.execute(s1)
    reveal_type(r1)

def t_from_statement() -> None:
    if False:
        print('Hello World!')
    t = text('select * from user')
    reveal_type(t)
    select(User).from_statement(t)
    ts = text('select * from user').columns(User.id, User.name)
    reveal_type(ts)
    select(User).from_statement(ts)
    ts2 = text('select * from user').columns(user_table.c.id, user_table.c.name)
    reveal_type(ts2)
    select(User).from_statement(ts2)

def t_aliased_fromclause() -> None:
    if False:
        i = 10
        return i + 15
    a1 = aliased(User, user_table)
    a2 = aliased(User, user_table.alias())
    a3 = aliased(User, join(user_table, user_table.alias()))
    a4 = aliased(user_table)
    reveal_type(a1)
    reveal_type(a2)
    reveal_type(a3)
    reveal_type(a4)

def test_select_from() -> None:
    if False:
        return 10
    select(1).select_from(User).exists()
    exists(1).select_from(User).select()
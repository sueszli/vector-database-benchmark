"""test sessionmaker, originally for #7656"""
from sqlalchemy import create_engine
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import QueryPropertyDescriptor
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
async_engine = create_async_engine('...')

class MyAsyncSession(AsyncSession):
    pass

def async_session_factory(engine: AsyncEngine) -> async_sessionmaker[MyAsyncSession]:
    if False:
        print('Hello World!')
    return async_sessionmaker(engine, class_=MyAsyncSession)

def async_scoped_session_factory(engine: AsyncEngine) -> async_scoped_session[MyAsyncSession]:
    if False:
        return 10
    return async_scoped_session(async_sessionmaker(engine, class_=MyAsyncSession), scopefunc=lambda : None)

async def async_main() -> None:
    fac = async_session_factory(async_engine)
    async with fac() as sess:
        reveal_type(sess)
    async with fac.begin() as sess:
        reveal_type(sess)
    scoped_fac = async_scoped_session_factory(async_engine)
    sess = scoped_fac()
    reveal_type(sess)
engine = create_engine('...')

class MySession(Session):
    pass

def session_factory(engine: Engine) -> sessionmaker[MySession]:
    if False:
        while True:
            i = 10
    return sessionmaker(engine, class_=MySession)

def scoped_session_factory(engine: Engine) -> scoped_session[MySession]:
    if False:
        i = 10
        return i + 15
    return scoped_session(sessionmaker(engine, class_=MySession))

def main() -> None:
    if False:
        while True:
            i = 10
    fac = session_factory(engine)
    with fac() as sess:
        reveal_type(sess)
    with fac.begin() as sess:
        reveal_type(sess)
    scoped_fac = scoped_session_factory(engine)
    sess = scoped_fac()
    reveal_type(sess)

def test_8837_sync() -> None:
    if False:
        for i in range(10):
            print('nop')
    sm = sessionmaker()
    reveal_type(sm)
    session = sm()
    reveal_type(session)

def test_8837_async() -> None:
    if False:
        while True:
            i = 10
    as_ = async_sessionmaker()
    reveal_type(as_)
    async_session = as_()
    reveal_type(async_session)
ss_9338 = scoped_session_factory(engine)
reveal_type(ss_9338.query_property())
qp: QueryPropertyDescriptor = ss_9338.query_property()

class Foo:
    query = qp
reveal_type(Foo.query)
reveal_type(Foo.query.all())

class Bar:
    query: QueryPropertyDescriptor = ss_9338.query_property()
reveal_type(Bar.query)
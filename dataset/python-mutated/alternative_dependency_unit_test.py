""" Service unit testing best practice, with an alternative dependency.
"""
import pytest
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from nameko.rpc import rpc
from nameko.testing.services import worker_factory
from nameko_sqlalchemy import Session
Base = declarative_base()

class Result(Base):
    __tablename__ = 'model'
    id = Column(Integer, primary_key=True)
    value = Column(String(64))

class Service:
    """ Service under test
    """
    name = 'service'
    db = Session(Base)

    @rpc
    def save(self, value):
        if False:
            print('Hello World!')
        result = Result(value=value)
        self.db.add(result)
        self.db.commit()

@pytest.fixture
def session():
    if False:
        for i in range(10):
            print('nop')
    ' Create a test database and session\n    '
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session_cls = sessionmaker(bind=engine)
    return session_cls()

def test_service(session):
    if False:
        while True:
            i = 10
    service = worker_factory(Service, db=session)
    service.save('helloworld')
    assert session.query(Result.value).all() == [('helloworld',)]
"""Illustrates the same UPDATE into INSERT technique of ``versioned_rows.py``,
but also emits an UPDATE on the **old** row to affect a change in timestamp.
Also includes a :meth:`.SessionEvents.do_orm_execute` hook to limit queries
to only the most recent version.

"""
import datetime
import time
from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import attributes
from sqlalchemy.orm import backref
from sqlalchemy.orm import make_transient
from sqlalchemy.orm import make_transient_to_detached
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.orm import with_loader_criteria
Base = declarative_base()
now = None

def current_time():
    if False:
        print('Hello World!')
    return now

class VersionedStartEnd:
    start = Column(DateTime, primary_key=True)
    end = Column(DateTime, primary_key=True)

    def __init__(self, **kw):
        if False:
            i = 10
            return i + 15
        kw.setdefault('start', current_time() - datetime.timedelta(days=3))
        kw.setdefault('end', current_time() + datetime.timedelta(days=3))
        super().__init__(**kw)

    def new_version(self, session):
        if False:
            while True:
                i = 10
        old_identity_key = inspect(self).key
        (self.id, self.start, self.end)
        make_transient(self)
        old_copy_of_us = self.__class__(id=self.id, start=self.start, end=self.end)
        make_transient_to_detached(old_copy_of_us)
        assert inspect(old_copy_of_us).key == old_identity_key
        session.add(old_copy_of_us)
        old_copy_of_us.end = current_time()
        self.start = current_time()
        self.end = current_time() + datetime.timedelta(days=2)

@event.listens_for(Session, 'before_flush')
def before_flush(session, flush_context, instances):
    if False:
        while True:
            i = 10
    for instance in session.dirty:
        if not isinstance(instance, VersionedStartEnd):
            continue
        if not session.is_modified(instance):
            continue
        if not attributes.instance_state(instance).has_identity:
            continue
        instance.new_version(session)
        session.add(instance)

@event.listens_for(Session, 'do_orm_execute', retval=True)
def do_orm_execute(execute_state):
    if False:
        i = 10
        return i + 15
    'ensure all queries for VersionedStartEnd include criteria'
    ct = current_time() + datetime.timedelta(seconds=1)
    execute_state.statement = execute_state.statement.options(with_loader_criteria(VersionedStartEnd, lambda cls: and_(ct > cls.start, ct < cls.end), include_aliases=True))

class Parent(VersionedStartEnd, Base):
    __tablename__ = 'parent'
    id = Column(Integer, primary_key=True)
    start = Column(DateTime, primary_key=True)
    end = Column(DateTime, primary_key=True)
    data = Column(String)
    child_n = Column(Integer)
    child = relationship('Child', primaryjoin='Child.id == foreign(Parent.child_n)', uselist=False, backref=backref('parent', uselist=False))

class Child(VersionedStartEnd, Base):
    __tablename__ = 'child'
    id = Column(Integer, primary_key=True)
    start = Column(DateTime, primary_key=True)
    end = Column(DateTime, primary_key=True)
    data = Column(String)

    def new_version(self, session):
        if False:
            for i in range(10):
                print('nop')
        session.expire(self.parent, ['child'])
        VersionedStartEnd.new_version(self, session)
        self.parent.child = self
times = []

def time_passes(s):
    if False:
        return 10
    'keep track of timestamps in terms of the database and allow time to\n    pass between steps.'
    s.commit()
    if times:
        time.sleep(1)
    times.append(datetime.datetime.now())
    if len(times) > 1:
        assert times[-1] > times[-2]
    return times[-1]
e = create_engine('sqlite://', echo='debug')
Base.metadata.create_all(e)
s = Session(e)
now = time_passes(s)
c1 = Child(id=1, data='child 1')
p1 = Parent(id=1, data='c1', child=c1)
s.add(p1)
s.commit()
assert s.query(Parent.__table__).all() == [(1, times[0] - datetime.timedelta(days=3), times[0] + datetime.timedelta(days=3), 'c1', 1)]
assert s.query(Child.__table__).all() == [(1, times[0] - datetime.timedelta(days=3), times[0] + datetime.timedelta(days=3), 'child 1')]
now = time_passes(s)
p1_check = s.query(Parent).first()
assert p1_check is p1
assert p1_check.child is c1
p1.child.data = 'elvis presley'
s.commit()
p2_check = s.query(Parent).first()
assert p2_check is p1_check
c2_check = p2_check.child
assert p2_check.child is c1
assert c1.data == 'elvis presley'
assert c1.end == now + datetime.timedelta(days=2)
assert s.query(Parent.__table__).all() == [(1, times[0] - datetime.timedelta(days=3), times[0] + datetime.timedelta(days=3), 'c1', 1)]
assert s.query(Child.__table__).order_by(Child.end).all() == [(1, times[0] - datetime.timedelta(days=3), times[1], 'child 1'), (1, times[1], times[1] + datetime.timedelta(days=2), 'elvis presley')]
now = time_passes(s)
p1.data = 'c2 elvis presley'
s.commit()
assert s.query(Parent.__table__).order_by(Parent.end).all() == [(1, times[0] - datetime.timedelta(days=3), times[2], 'c1', 1), (1, times[2], times[2] + datetime.timedelta(days=2), 'c2 elvis presley', 1)]
assert s.query(Child.__table__).order_by(Child.end).all() == [(1, times[0] - datetime.timedelta(days=3), times[1], 'child 1'), (1, times[1], times[1] + datetime.timedelta(days=2), 'elvis presley')]
s.add(Parent(id=2, data='unrelated', child=Child(id=2, data='unrelated')))
s.commit()
p3_check = s.query(Parent).filter_by(id=1).one()
assert p3_check is p1
assert p3_check.child is c1
c3_check = s.query(Child).filter(Child.parent == p3_check).one()
assert c3_check is c1
c3_check = s.query(Child).join(Parent.child).filter(Parent.id == p3_check.id).one()
"""Illustrates a global criteria applied to entities of a particular type.

The example here is the "public" flag, a simple boolean that indicates
the rows are part of a publicly viewable subcategory.  Rows that do not
include this flag are not shown unless a special option is passed to the
query.

Uses for this kind of recipe include tables that have "soft deleted" rows
marked as "deleted" that should be skipped, rows that have access control rules
that should be applied on a per-request basis, etc.


"""
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import true
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

@event.listens_for(Session, 'do_orm_execute')
def _add_filtering_criteria(execute_state):
    if False:
        while True:
            i = 10
    'Intercept all ORM queries.   Add a with_loader_criteria option to all\n    of them.\n\n    This option applies to SELECT queries and adds a global WHERE criteria\n    (or as appropriate ON CLAUSE criteria for join targets)\n    to all objects of a certain class or superclass.\n\n    '
    if not execute_state.is_column_load and (not execute_state.is_relationship_load) and (not execute_state.execution_options.get('include_private', False)):
        execute_state.statement = execute_state.statement.options(orm.with_loader_criteria(HasPrivate, lambda cls: cls.public == true(), include_aliases=True))

class HasPrivate:
    """Mixin that identifies a class as having private entities"""
    public = Column(Boolean, nullable=False)
if __name__ == '__main__':
    Base = declarative_base()

    class User(HasPrivate, Base):
        __tablename__ = 'user'
        id = Column(Integer, primary_key=True)
        name = Column(String)
        addresses = relationship('Address', back_populates='user')

    class Address(HasPrivate, Base):
        __tablename__ = 'address'
        id = Column(Integer, primary_key=True)
        email = Column(String)
        user_id = Column(Integer, ForeignKey('user.id'))
        user = relationship('User', back_populates='addresses')
    engine = create_engine('sqlite://', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    sess.add_all([User(name='u1', public=True, addresses=[Address(email='u1a1', public=True), Address(email='u1a2', public=True)]), User(name='u2', public=True, addresses=[Address(email='u2a1', public=False), Address(email='u2a2', public=True)]), User(name='u3', public=False, addresses=[Address(email='u3a1', public=False), Address(email='u3a2', public=False)]), User(name='u4', public=False, addresses=[Address(email='u4a1', public=False), Address(email='u4a2', public=True)]), User(name='u5', public=True, addresses=[Address(email='u5a1', public=True), Address(email='u5a2', public=False)])])
    sess.commit()
    for u1 in sess.query(User).options(orm.selectinload(User.addresses)):
        assert u1.public
        for address in u1.addresses:
            assert address.public
    cols = sess.query(User.id, Address.id).join(User.addresses).order_by(User.id, Address.id).all()
    assert cols == [(1, 1), (1, 2), (2, 4), (5, 9)]
    cols = sess.query(User.id, Address.id).join(User.addresses).order_by(User.id, Address.id).execution_options(include_private=True).all()
    assert cols == [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (4, 7), (4, 8), (5, 9), (5, 10)]
    assert sess.query(Address).count() == 5
    assert sess.query(Address).execution_options(include_private=True).count() == 10
    a1 = sess.execute(select(Address).filter_by(email='u4a2')).scalar()
    assert a1.user is None
    sess.expire(a1, ['user'])
    u1 = sess.execute(select(User).filter_by(name='u4').execution_options(include_private=True)).scalar()
    assert a1.user is u1
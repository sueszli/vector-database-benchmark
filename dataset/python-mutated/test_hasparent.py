"""test the current state of the hasparent() flag."""
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import testing
from sqlalchemy.orm import attributes
from sqlalchemy.orm import exc as orm_exc
from sqlalchemy.orm import relationship
from sqlalchemy.testing import assert_raises_message
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table
from sqlalchemy.testing.util import gc_collect

class ParentRemovalTest(fixtures.MappedTest):
    """Test that the 'hasparent' flag gets flipped to False
    only if we're sure this object is the real parent.

    In ambiguous cases a stale data exception is
    raised.

    """
    run_inserts = None
    run_setup_classes = 'each'
    run_setup_mappers = 'each'

    @classmethod
    def define_tables(cls, metadata):
        if False:
            for i in range(10):
                print('nop')
        if testing.against('oracle'):
            fk_args = dict(deferrable=True, initially='deferred')
        elif testing.against('mysql'):
            fk_args = {}
        else:
            fk_args = dict(onupdate='cascade')
        Table('users', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True))
        Table('addresses', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('user_id', Integer, ForeignKey('users.id', **fk_args)))

    @classmethod
    def setup_classes(cls):
        if False:
            return 10

        class User(cls.Comparable):
            pass

        class Address(cls.Comparable):
            pass

    @classmethod
    def setup_mappers(cls):
        if False:
            print('Hello World!')
        cls.mapper_registry.map_imperatively(cls.classes.Address, cls.tables.addresses)
        cls.mapper_registry.map_imperatively(cls.classes.User, cls.tables.users, properties={'addresses': relationship(cls.classes.Address, cascade='all, delete-orphan')})

    def _assert_hasparent(self, a1):
        if False:
            for i in range(10):
                print('nop')
        assert attributes.has_parent(self.classes.User, a1, 'addresses')

    def _assert_not_hasparent(self, a1):
        if False:
            print('Hello World!')
        assert not attributes.has_parent(self.classes.User, a1, 'addresses')

    def _fixture(self):
        if False:
            return 10
        (User, Address) = (self.classes.User, self.classes.Address)
        s = fixture_session()
        u1 = User()
        a1 = Address()
        u1.addresses.append(a1)
        s.add(u1)
        s.flush()
        return (s, u1, a1)

    def test_stale_state_positive(self):
        if False:
            print('Hello World!')
        User = self.classes.User
        (s, u1, a1) = self._fixture()
        s.expunge(u1)
        u1 = s.query(User).first()
        u1.addresses.remove(a1)
        self._assert_not_hasparent(a1)

    @testing.requires.predictable_gc
    def test_stale_state_positive_gc(self):
        if False:
            i = 10
            return i + 15
        User = self.classes.User
        (s, u1, a1) = self._fixture()
        s.expunge(u1)
        del u1
        gc_collect()
        u1 = s.query(User).first()
        u1.addresses.remove(a1)
        self._assert_not_hasparent(a1)

    @testing.requires.updateable_autoincrement_pks
    @testing.requires.predictable_gc
    def test_stale_state_positive_pk_change(self):
        if False:
            for i in range(10):
                print('nop')
        "Illustrate that we can't easily link a\n        stale state to a fresh one if the fresh one has\n        a PK change  (unless we a. tracked all the previous PKs,\n        wasteful, or b. recycled states - time consuming,\n        breaks lots of edge cases, destabilizes the code)\n\n        "
        User = self.classes.User
        (s, u1, a1) = self._fixture()
        s._expunge_states([attributes.instance_state(u1)])
        del u1
        gc_collect()
        u1 = s.query(User).first()
        new_id = u1.id + 10
        u1.id = new_id
        a1.user_id = new_id
        s.flush()
        assert_raises_message(orm_exc.StaleDataError, "can't be sure this is the most recent parent.", u1.addresses.remove, a1)
        eq_(u1.addresses, [a1])
        s.expire_all()
        u1.addresses.remove(a1)
        self._assert_not_hasparent(a1)

    def test_stale_state_negative_child_expired(self):
        if False:
            print('Hello World!')
        "illustrate the current behavior of\n        expiration on the child.\n\n        there's some uncertainty here in how\n        this use case should work.\n\n        "
        User = self.classes.User
        (s, u1, a1) = self._fixture()
        gc_collect()
        u2 = User(addresses=[a1])
        s.expire(a1)
        u1.addresses.remove(a1)
        u2_is = u2._sa_instance_state
        del u2
        for i in range(5):
            gc_collect()
        o = u2_is.obj()
        assert o is None
        self._assert_not_hasparent(a1)

    @testing.requires.predictable_gc
    def test_stale_state_negative(self):
        if False:
            while True:
                i = 10
        User = self.classes.User
        (s, u1, a1) = self._fixture()
        gc_collect()
        u2 = User(addresses=[a1])
        s.add(u2)
        s.flush()
        s._expunge_states([attributes.instance_state(u2)])
        u2_is = u2._sa_instance_state
        del u2
        for i in range(5):
            gc_collect()
        o = u2_is.obj()
        assert o is None
        assert_raises_message(orm_exc.StaleDataError, "can't be sure this is the most recent parent.", u1.addresses.remove, a1)
        s.flush()
        self._assert_hasparent(a1)

    def test_fresh_state_positive(self):
        if False:
            for i in range(10):
                print('nop')
        (s, u1, a1) = self._fixture()
        self._assert_hasparent(a1)

    def test_fresh_state_negative(self):
        if False:
            return 10
        (s, u1, a1) = self._fixture()
        u1.addresses.remove(a1)
        self._assert_not_hasparent(a1)
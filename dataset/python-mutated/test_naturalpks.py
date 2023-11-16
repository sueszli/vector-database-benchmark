"""
Primary key changing capabilities and passive/non-passive cascading updates.

"""
import itertools
import sqlalchemy as sa
from sqlalchemy import bindparam
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import testing
from sqlalchemy import TypeDecorator
from sqlalchemy.orm import make_transient
from sqlalchemy.orm import relationship
from sqlalchemy.orm import with_parent
from sqlalchemy.testing import assert_raises
from sqlalchemy.testing import assert_raises_message
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_warnings
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import ne_
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table
from test.orm import _fixtures

def _backend_specific_fk_args():
    if False:
        while True:
            i = 10
    if testing.requires.deferrable_fks.enabled and testing.requires.non_updating_cascade.enabled:
        fk_args = dict(deferrable=True, initially='deferred')
    elif not testing.requires.on_update_cascade.enabled:
        fk_args = dict()
    else:
        fk_args = dict(onupdate='cascade')
    return fk_args

class NaturalPKTest(fixtures.MappedTest):
    __requires__ = ('skip_mysql_on_windows',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            while True:
                i = 10
        fk_args = _backend_specific_fk_args()
        Table('users', metadata, Column('username', String(50), primary_key=True), Column('fullname', String(100)), test_needs_fk=True)
        Table('addresses', metadata, Column('email', String(50), primary_key=True), Column('username', String(50), ForeignKey('users.username', **fk_args)), test_needs_fk=True)
        Table('items', metadata, Column('itemname', String(50), primary_key=True), Column('description', String(100)), test_needs_fk=True)
        Table('users_to_items', metadata, Column('username', String(50), ForeignKey('users.username', **fk_args), primary_key=True), Column('itemname', String(50), ForeignKey('items.itemname', **fk_args), primary_key=True), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            return 10

        class User(cls.Comparable):
            pass

        class Address(cls.Comparable):
            pass

        class Item(cls.Comparable):
            pass

    def test_entity(self):
        if False:
            print('Hello World!')
        (users, User) = (self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        assert sess.get(User, 'jack') is u1
        u1.username = 'ed'
        sess.flush()

        def go():
            if False:
                print('Hello World!')
            assert sess.get(User, 'ed') is u1
        self.assert_sql_count(testing.db, go, 0)
        assert sess.get(User, 'jack') is None
        sess.expunge_all()
        u1 = sess.get(User, 'ed')
        eq_(User(username='ed', fullname='jack'), u1)

    def test_load_after_expire(self):
        if False:
            i = 10
            return i + 15
        (users, User) = (self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        assert sess.get(User, 'jack') is u1
        sess.execute(users.update().values({User.username: 'jack'}), dict(username='ed'))
        sess.expire(u1)
        assert_raises(sa.orm.exc.ObjectDeletedError, getattr, u1, 'username')
        sess.expunge_all()
        assert sess.get(User, 'jack') is None
        assert sess.get(User, 'ed').fullname == 'jack'

    @testing.requires.update_returning
    def test_update_to_sql_expr(self):
        if False:
            while True:
                i = 10
        (users, User) = (self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        u1.username = User.username + ' jones'
        sess.flush()
        eq_(u1.username, 'jack jones')

    def test_update_to_self_sql_expr(self):
        if False:
            while True:
                i = 10
        (users, User) = (self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        u1.username = User.username + ''
        sess.flush()
        eq_(u1.username, 'jack')

    def test_flush_new_pk_after_expire(self):
        if False:
            i = 10
            return i + 15
        (User, users) = (self.classes.User, self.tables.users)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        assert sess.get(User, 'jack') is u1
        sess.expire(u1)
        u1.username = 'ed'
        sess.flush()
        sess.expunge_all()
        assert sess.get(User, 'ed').fullname == 'jack'

    @testing.requires.on_update_cascade
    def test_onetomany_passive(self):
        if False:
            return 10
        self._test_onetomany(True)

    def test_onetomany_nonpassive(self):
        if False:
            return 10
        self._test_onetomany(False)

    def _test_onetomany(self, passive_updates):
        if False:
            while True:
                i = 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        u1.addresses.append(Address(email='jack1'))
        u1.addresses.append(Address(email='jack2'))
        sess.add(u1)
        sess.flush()
        assert sess.get(Address, 'jack1') is u1.addresses[0]
        u1.username = 'ed'
        sess.flush()
        assert u1.addresses[0].username == 'ed'
        sess.expunge_all()
        eq_([Address(username='ed'), Address(username='ed')], sess.query(Address).all())
        u1 = sess.get(User, 'ed')
        u1.username = 'jack'

        def go():
            if False:
                while True:
                    i = 10
            sess.flush()
        if not passive_updates:
            self.assert_sql_count(testing.db, go, 3)
        else:
            self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()
        assert User(username='jack', addresses=[Address(username='jack'), Address(username='jack')]) == sess.get(User, 'jack')
        u1 = sess.get(User, 'jack')
        u1.addresses = []
        u1.username = 'fred'
        sess.flush()
        sess.expunge_all()
        assert sess.get(Address, 'jack1').username is None
        u1 = sess.get(User, 'fred')
        eq_(User(username='fred', fullname='jack'), u1)

    @testing.requires.on_update_cascade
    def test_manytoone_passive(self):
        if False:
            print('Hello World!')
        self._test_manytoone(True)

    def test_manytoone_nonpassive(self):
        if False:
            i = 10
            return i + 15
        self._test_manytoone(False)

    @testing.requires.on_update_cascade
    def test_manytoone_passive_uselist(self):
        if False:
            while True:
                i = 10
        self._test_manytoone(True, True)

    def test_manytoone_nonpassive_uselist(self):
        if False:
            return 10
        self._test_manytoone(False, True)

    def test_manytoone_nonpassive_cold_mapping(self):
        if False:
            return 10
        "test that the mapper-level m2o dependency processor\n        is set up even if the opposite side relationship\n        hasn't yet been part of a flush.\n\n        "
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        with testing.db.begin() as conn:
            conn.execute(users.insert(), dict(username='jack', fullname='jack'))
            conn.execute(addresses.insert(), dict(email='jack1', username='jack'))
            conn.execute(addresses.insert(), dict(email='jack2', username='jack'))
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, passive_updates=False)})
        sess = fixture_session()
        u1 = sess.query(User).first()
        (a1, a2) = sess.query(Address).all()
        u1.username = 'ed'

        def go():
            if False:
                i = 10
                return i + 15
            sess.flush()
        self.assert_sql_count(testing.db, go, 2)

    def _test_manytoone(self, passive_updates, uselist=False, dynamic=False):
        if False:
            return 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, uselist=uselist, passive_updates=passive_updates)})
        sess = fixture_session()
        a1 = Address(email='jack1')
        a2 = Address(email='jack2')
        a3 = Address(email='fred')
        u1 = User(username='jack', fullname='jack')
        if uselist:
            a1.user = [u1]
            a2.user = [u1]
        else:
            a1.user = u1
            a2.user = u1
        sess.add(a1)
        sess.add(a2)
        sess.add(a3)
        sess.flush()
        u1.username = 'ed'

        def go():
            if False:
                print('Hello World!')
            sess.flush()
        if passive_updates:
            self.assert_sql_count(testing.db, go, 1)
        else:
            self.assert_sql_count(testing.db, go, 2)

        def go():
            if False:
                print('Hello World!')
            sess.flush()
        self.assert_sql_count(testing.db, go, 0)
        assert a1.username == a2.username == 'ed'
        sess.expunge_all()
        if uselist:
            eq_([Address(email='fred', user=[]), Address(username='ed'), Address(username='ed')], sess.query(Address).order_by(Address.email).all())
        else:
            eq_([Address(email='fred', user=None), Address(username='ed'), Address(username='ed')], sess.query(Address).order_by(Address.email).all())

    @testing.requires.on_update_cascade
    def test_onetoone_passive(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_onetoone(True)

    def test_onetoone_nonpassive(self):
        if False:
            while True:
                i = 10
        self._test_onetoone(False)

    def _test_onetoone(self, passive_updates):
        if False:
            return 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'address': relationship(Address, passive_updates=passive_updates, uselist=False)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        sess.add(u1)
        sess.flush()
        a1 = Address(email='jack1')
        u1.address = a1
        sess.add(a1)
        sess.flush()
        u1.username = 'ed'

        def go():
            if False:
                for i in range(10):
                    print('nop')
            sess.flush()
        if passive_updates:
            sess.expire(u1, ['address'])
            self.assert_sql_count(testing.db, go, 1)
        else:
            self.assert_sql_count(testing.db, go, 2)

        def go():
            if False:
                while True:
                    i = 10
            sess.flush()
        self.assert_sql_count(testing.db, go, 0)
        sess.expunge_all()
        eq_([Address(username='ed')], sess.query(Address).all())

    @testing.requires.on_update_cascade
    def test_bidirectional_passive(self):
        if False:
            while True:
                i = 10
        self._test_bidirectional(True)

    def test_bidirectional_nonpassive(self):
        if False:
            return 10
        self._test_bidirectional(False)

    def _test_bidirectional(self, passive_updates):
        if False:
            i = 10
            return i + 15
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, passive_updates=passive_updates, backref='addresses')})
        sess = fixture_session(autoflush=False)
        a1 = Address(email='jack1')
        a2 = Address(email='jack2')
        u1 = User(username='jack', fullname='jack')
        a1.user = u1
        a2.user = u1
        sess.add(a1)
        sess.add(a2)
        sess.flush()
        u1.username = 'ed'
        (ad1, ad2) = sess.query(Address).all()
        eq_([Address(username='jack'), Address(username='jack')], [ad1, ad2])

        def go():
            if False:
                while True:
                    i = 10
            sess.flush()
        if passive_updates:
            self.assert_sql_count(testing.db, go, 1)
        else:
            self.assert_sql_count(testing.db, go, 2)
        eq_([Address(username='ed'), Address(username='ed')], [ad1, ad2])
        sess.expunge_all()
        eq_([Address(username='ed'), Address(username='ed')], sess.query(Address).all())
        u1 = sess.get(User, 'ed')
        assert len(u1.addresses) == 2
        u1.username = 'fred'

        def go():
            if False:
                return 10
            sess.flush()
        if passive_updates:
            self.assert_sql_count(testing.db, go, 1)
        else:
            self.assert_sql_count(testing.db, go, 2)
        sess.expunge_all()
        eq_([Address(username='fred'), Address(username='fred')], sess.query(Address).all())

    @testing.requires.on_update_cascade
    def test_manytomany_passive(self):
        if False:
            while True:
                i = 10
        self._test_manytomany(True)

    @testing.fails_if(testing.requires.on_update_cascade + testing.requires.sane_multi_rowcount)
    def test_manytomany_nonpassive(self):
        if False:
            return 10
        self._test_manytomany(False)

    def _test_manytomany(self, passive_updates):
        if False:
            print('Hello World!')
        (users, items, Item, User, users_to_items) = (self.tables.users, self.tables.items, self.classes.Item, self.classes.User, self.tables.users_to_items)
        self.mapper_registry.map_imperatively(User, users, properties={'items': relationship(Item, secondary=users_to_items, backref='users', passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Item, items)
        sess = fixture_session()
        u1 = User(username='jack')
        u2 = User(username='fred')
        i1 = Item(itemname='item1')
        i2 = Item(itemname='item2')
        u1.items.append(i1)
        u1.items.append(i2)
        i2.users.append(u2)
        sess.add(u1)
        sess.add(u2)
        sess.flush()
        r = sess.query(Item).all()
        eq_(Item(itemname='item1'), r[0])
        eq_(['jack'], [u.username for u in r[0].users])
        eq_(Item(itemname='item2'), r[1])
        eq_(['jack', 'fred'], [u.username for u in r[1].users])
        u2.username = 'ed'

        def go():
            if False:
                while True:
                    i = 10
            sess.flush()
        go()

        def go():
            if False:
                print('Hello World!')
            sess.flush()
        self.assert_sql_count(testing.db, go, 0)
        sess.expunge_all()
        r = sess.query(Item).all()
        eq_(Item(itemname='item1'), r[0])
        eq_(['jack'], [u.username for u in r[0].users])
        eq_(Item(itemname='item2'), r[1])
        eq_(['ed', 'jack'], sorted([u.username for u in r[1].users]))
        sess.expunge_all()
        u2 = sess.get(User, u2.username)
        u2.username = 'wendy'
        sess.flush()
        r = sess.query(Item).filter(with_parent(u2, User.items)).all()
        eq_(Item(itemname='item2'), r[0])

    def test_manytoone_deferred_relationship_expr(self):
        if False:
            for i in range(10):
                print('nop')
        'for [ticket:4359], test that updates to the columns embedded\n        in an object expression are also updated.'
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, passive_updates=testing.requires.on_update_cascade.enabled)})
        s = fixture_session()
        a1 = Address(email='jack1')
        u1 = User(username='jack', fullname='jack')
        a1.user = u1
        expr = Address.user == u1
        eq_(expr.left.callable(), 'jack')
        u1.username = 'ed'
        eq_(expr.left.callable(), 'ed')
        s.add_all([u1, a1])
        s.commit()
        eq_(a1.username, 'ed')
        u1.username = 'fred'
        s.flush()
        eq_(expr.left.callable(), 'fred')
        u1.username = 'wendy'
        s.commit()
        assert 'username' not in u1.__dict__
        eq_(expr.left.callable(), 'wendy')
        u1.username = 'jack'
        s.commit()
        assert 'username' not in u1.__dict__
        s.expunge(u1)
        eq_(expr.left.callable(), 'jack')
        assert 'username' not in u1.__dict__
        s.add(u1)
        eq_(expr.left.callable(), 'jack')
        assert 'username' in u1.__dict__
        u2 = User(username='jack', fullname='jack')
        expr = Address.user == u2
        eq_(expr.left.callable(), 'jack')
        del u2.username
        assert_raises_message(sa.exc.InvalidRequestError, "Can't resolve value for column users.username", expr.left.callable)
        u2.username = 'ed'
        eq_(expr.left.callable(), 'ed')
        s.add(u2)
        s.commit()
        eq_(expr.left.callable(), 'ed')
        del u2.username
        with expect_warnings('Got None for value of column '):
            eq_(expr.left.callable(), None)
        s.expunge(u2)
        assert 'username' not in u2.__dict__
        assert_raises_message(sa.exc.InvalidRequestError, "Can't resolve value for column users.username", expr.left.callable)

class TransientExceptionTesst(_fixtures.FixtureTest):
    run_inserts = None
    __backend__ = True

    def test_transient_exception(self):
        if False:
            while True:
                i = 10
        'An object that goes from a pk value to transient/pending\n        doesn\'t count as a "pk" switch.\n\n        '
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User)})
        sess = fixture_session()
        u1 = User(id=5, name='u1')
        ad1 = Address(email_address='e1', user=u1)
        sess.add_all([u1, ad1])
        sess.flush()
        make_transient(u1)
        u1.id = None
        u1.username = 'u2'
        sess.add(u1)
        sess.flush()
        eq_(ad1.user_id, 5)
        sess.expire_all()
        eq_(ad1.user_id, 5)
        ne_(u1.id, 5)
        ne_(u1.id, None)
        eq_(sess.query(User).count(), 2)

class ReversePKsTest(fixtures.MappedTest):
    """reverse the primary keys of two entities and ensure bookkeeping
    succeeds."""
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            i = 10
            return i + 15
        Table('user', metadata, Column('code', Integer, autoincrement=False, primary_key=True), Column('status', Integer, autoincrement=False, primary_key=True), Column('username', String(50), nullable=False), test_needs_acid=True)

    @classmethod
    def setup_classes(cls):
        if False:
            while True:
                i = 10

        class User(cls.Comparable):

            def __init__(self, code, status, username):
                if False:
                    print('Hello World!')
                self.code = code
                self.status = status
                self.username = username

    def test_reverse(self):
        if False:
            print('Hello World!')
        (user, User) = (self.tables.user, self.classes.User)
        (PUBLISHED, EDITABLE, ARCHIVED) = (1, 2, 3)
        self.mapper_registry.map_imperatively(User, user)
        session = fixture_session()
        a_published = User(1, PUBLISHED, 'a')
        session.add(a_published)
        session.commit()
        a_editable = User(1, EDITABLE, 'a')
        session.add(a_editable)
        session.commit()
        a_published.status = ARCHIVED
        a_editable.status = PUBLISHED
        session.commit()
        assert session.get(User, [1, PUBLISHED]) is a_editable
        assert session.get(User, [1, ARCHIVED]) is a_published
        a_published.status = PUBLISHED
        a_editable.status = EDITABLE
        session.commit()
        assert session.get(User, [1, PUBLISHED]) is a_published
        assert session.get(User, [1, EDITABLE]) is a_editable

    @testing.requires.savepoints
    def test_reverse_savepoint(self):
        if False:
            for i in range(10):
                print('nop')
        (user, User) = (self.tables.user, self.classes.User)
        (PUBLISHED, EDITABLE, ARCHIVED) = (1, 2, 3)
        self.mapper_registry.map_imperatively(User, user)
        session = fixture_session()
        a_published = User(1, PUBLISHED, 'a')
        session.add(a_published)
        session.commit()
        a_editable = User(1, EDITABLE, 'a')
        session.add(a_editable)
        session.commit()
        nt1 = session.begin_nested()
        a_published.status = ARCHIVED
        a_editable.status = PUBLISHED
        nt1.commit()
        session.rollback()
        eq_(a_published.status, PUBLISHED)
        eq_(a_editable.status, EDITABLE)

class SelfReferentialTest(fixtures.MappedTest):
    __unsupported_on__ = ('mssql', 'mysql', 'mariadb')
    __requires__ = ('on_update_or_deferrable_fks',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            i = 10
            return i + 15
        fk_args = _backend_specific_fk_args()
        Table('nodes', metadata, Column('name', String(50), primary_key=True), Column('parent', String(50), ForeignKey('nodes.name', **fk_args)), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            while True:
                i = 10

        class Node(cls.Comparable):
            pass

    def test_one_to_many_on_m2o(self):
        if False:
            return 10
        (Node, nodes) = (self.classes.Node, self.tables.nodes)
        self.mapper_registry.map_imperatively(Node, nodes, properties={'children': relationship(Node, backref=sa.orm.backref('parentnode', remote_side=nodes.c.name, passive_updates=False))})
        sess = fixture_session(future=True)
        n1 = Node(name='n1')
        sess.add(n1)
        n2 = Node(name='n11', parentnode=n1)
        n3 = Node(name='n12', parentnode=n1)
        n4 = Node(name='n13', parentnode=n1)
        sess.add_all([n2, n3, n4])
        sess.commit()
        n1.name = 'new n1'
        sess.commit()
        eq_(['new n1', 'new n1', 'new n1'], [n.parent for n in sess.query(Node).filter(Node.name.in_(['n11', 'n12', 'n13']))])

    def test_one_to_many_on_o2m(self):
        if False:
            i = 10
            return i + 15
        (Node, nodes) = (self.classes.Node, self.tables.nodes)
        self.mapper_registry.map_imperatively(Node, nodes, properties={'children': relationship(Node, backref=sa.orm.backref('parentnode', remote_side=nodes.c.name), passive_updates=False)})
        sess = fixture_session()
        n1 = Node(name='n1')
        n1.children.append(Node(name='n11'))
        n1.children.append(Node(name='n12'))
        n1.children.append(Node(name='n13'))
        sess.add(n1)
        sess.commit()
        n1.name = 'new n1'
        sess.commit()
        eq_(n1.children[1].parent, 'new n1')
        eq_(['new n1', 'new n1', 'new n1'], [n.parent for n in sess.query(Node).filter(Node.name.in_(['n11', 'n12', 'n13']))])

    @testing.requires.on_update_cascade
    def test_many_to_one_passive(self):
        if False:
            while True:
                i = 10
        self._test_many_to_one(True)

    def test_many_to_one_nonpassive(self):
        if False:
            return 10
        self._test_many_to_one(False)

    def _test_many_to_one(self, passive):
        if False:
            while True:
                i = 10
        (Node, nodes) = (self.classes.Node, self.tables.nodes)
        self.mapper_registry.map_imperatively(Node, nodes, properties={'parentnode': relationship(Node, remote_side=nodes.c.name, passive_updates=passive)})
        sess = fixture_session()
        n1 = Node(name='n1')
        n11 = Node(name='n11', parentnode=n1)
        n12 = Node(name='n12', parentnode=n1)
        n13 = Node(name='n13', parentnode=n1)
        sess.add_all([n1, n11, n12, n13])
        sess.commit()
        n1.name = 'new n1'
        sess.commit()
        eq_(['new n1', 'new n1', 'new n1'], [n.parent for n in sess.query(Node).filter(Node.name.in_(['n11', 'n12', 'n13']))])

class NonPKCascadeTest(fixtures.MappedTest):
    __requires__ = ('skip_mysql_on_windows', 'on_update_or_deferrable_fks')
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            return 10
        fk_args = _backend_specific_fk_args()
        Table('users', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('username', String(50), unique=True), Column('fullname', String(100)), test_needs_fk=True)
        Table('addresses', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('email', String(50)), Column('username', String(50), ForeignKey('users.username', **fk_args)), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            while True:
                i = 10

        class User(cls.Comparable):
            pass

        class Address(cls.Comparable):
            pass

    @testing.requires.on_update_cascade
    def test_onetomany_passive(self):
        if False:
            return 10
        self._test_onetomany(True)

    def test_onetomany_nonpassive(self):
        if False:
            print('Hello World!')
        self._test_onetomany(False)

    def _test_onetomany(self, passive_updates):
        if False:
            print('Hello World!')
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session()
        u1 = User(username='jack', fullname='jack')
        u1.addresses.append(Address(email='jack1'))
        u1.addresses.append(Address(email='jack2'))
        sess.add(u1)
        sess.flush()
        a1 = u1.addresses[0]
        eq_(sess.execute(sa.select(addresses.c.username)).fetchall(), [('jack',), ('jack',)])
        assert sess.get(Address, a1.id) is u1.addresses[0]
        u1.username = 'ed'
        sess.flush()
        assert u1.addresses[0].username == 'ed'
        eq_(sess.execute(sa.select(addresses.c.username)).fetchall(), [('ed',), ('ed',)])
        sess.expunge_all()
        eq_([Address(username='ed'), Address(username='ed')], sess.query(Address).all())
        u1 = sess.get(User, u1.id)
        u1.username = 'jack'

        def go():
            if False:
                return 10
            sess.flush()
        if not passive_updates:
            self.assert_sql_count(testing.db, go, 3)
        else:
            self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()
        assert User(username='jack', addresses=[Address(username='jack'), Address(username='jack')]) == sess.get(User, u1.id)
        sess.expunge_all()
        u1 = sess.get(User, u1.id)
        u1.addresses = []
        u1.username = 'fred'
        sess.flush()
        sess.expunge_all()
        a1 = sess.get(Address, a1.id)
        eq_(a1.username, None)
        eq_(sess.execute(sa.select(addresses.c.username)).fetchall(), [(None,), (None,)])
        u1 = sess.get(User, u1.id)
        eq_(User(username='fred', fullname='jack'), u1)

class CascadeToFKPKTest(fixtures.MappedTest, testing.AssertsCompiledSQL):
    """A primary key mutation cascades onto a foreign key that is itself a
    primary key."""
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            return 10
        fk_args = _backend_specific_fk_args()
        Table('users', metadata, Column('username', String(50), primary_key=True), test_needs_fk=True)
        Table('addresses', metadata, Column('username', String(50), ForeignKey('users.username', **fk_args), primary_key=True), Column('email', String(50), primary_key=True), Column('etc', String(50)), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')

        class User(cls.Comparable):
            pass

        class Address(cls.Comparable):
            pass

    @testing.requires.on_update_cascade
    def test_onetomany_passive(self):
        if False:
            i = 10
            return i + 15
        self._test_onetomany(True)

    @testing.requires.non_updating_cascade
    def test_onetomany_nonpassive(self):
        if False:
            i = 10
            return i + 15
        self._test_onetomany(False)

    def test_o2m_change_passive(self):
        if False:
            while True:
                i = 10
        self._test_o2m_change(True)

    def test_o2m_change_nonpassive(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_o2m_change(False)

    def _test_o2m_change(self, passive_updates):
        if False:
            for i in range(10):
                print('nop')
        'Change the PK of a related entity to another.\n\n        "on update cascade" is not involved here, so the mapper has\n        to do the UPDATE itself.\n\n        '
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session()
        a1 = Address(username='ed', email='ed@host1')
        u1 = User(username='ed', addresses=[a1])
        u2 = User(username='jack')
        sess.add_all([a1, u1, u2])
        sess.flush()
        a1.username = 'jack'
        sess.flush()

    def test_o2m_move_passive(self):
        if False:
            while True:
                i = 10
        self._test_o2m_move(True)

    def test_o2m_move_nonpassive(self):
        if False:
            i = 10
            return i + 15
        self._test_o2m_move(False)

    def _test_o2m_move(self, passive_updates):
        if False:
            i = 10
            return i + 15
        'Move the related entity to a different collection,\n        changing its PK.\n\n        '
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session(autoflush=False)
        a1 = Address(username='ed', email='ed@host1')
        u1 = User(username='ed', addresses=[a1])
        u2 = User(username='jack')
        sess.add_all([a1, u1, u2])
        sess.flush()
        u1.addresses.remove(a1)
        u2.addresses.append(a1)
        sess.flush()

    @testing.requires.on_update_cascade
    def test_change_m2o_passive(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_change_m2o(True)

    @testing.requires.non_updating_cascade
    def test_change_m2o_nonpassive(self):
        if False:
            return 10
        self._test_change_m2o(False)

    @testing.requires.on_update_cascade
    def test_change_m2o_passive_uselist(self):
        if False:
            i = 10
            return i + 15
        self._test_change_m2o(True, True)

    @testing.requires.non_updating_cascade
    def test_change_m2o_nonpassive_uselist(self):
        if False:
            return 10
        self._test_change_m2o(False, True)

    def _test_change_m2o(self, passive_updates, uselist=False):
        if False:
            return 10
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, uselist=uselist, passive_updates=passive_updates)})
        sess = fixture_session()
        u1 = User(username='jack')
        if uselist:
            a1 = Address(user=[u1], email='foo@bar')
        else:
            a1 = Address(user=u1, email='foo@bar')
        sess.add_all([u1, a1])
        sess.flush()
        u1.username = 'edmodified'
        sess.flush()
        eq_(a1.username, 'edmodified')
        sess.expire_all()
        eq_(a1.username, 'edmodified')

    def test_move_m2o_passive(self):
        if False:
            while True:
                i = 10
        self._test_move_m2o(True)

    def test_move_m2o_nonpassive(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_move_m2o(False)

    def _test_move_m2o(self, passive_updates):
        if False:
            while True:
                i = 10
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, passive_updates=passive_updates)})
        sess = fixture_session()
        u1 = User(username='jack')
        u2 = User(username='ed')
        a1 = Address(user=u1, email='foo@bar')
        sess.add_all([u1, u2, a1])
        sess.flush()
        a1.user = u2
        sess.flush()

    def test_rowswitch_doesntfire(self):
        if False:
            for i in range(10):
                print('nop')
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, passive_updates=True)})
        sess = fixture_session()
        u1 = User(username='ed')
        a1 = Address(user=u1, email='ed@host1')
        sess.add(u1)
        sess.add(a1)
        sess.flush()
        sess.delete(u1)
        sess.delete(a1)
        u2 = User(username='ed')
        a2 = Address(user=u2, email='ed@host1', etc='foo')
        sess.add(u2)
        sess.add(a2)
        from sqlalchemy.testing.assertsql import CompiledSQL
        self.assert_sql_execution(testing.db, sess.flush, CompiledSQL('UPDATE addresses SET etc=:etc WHERE addresses.username = :addresses_username AND addresses.email = :addresses_email', {'etc': 'foo', 'addresses_username': 'ed', 'addresses_email': 'ed@host1'}))

    def _test_onetomany(self, passive_updates):
        if False:
            for i in range(10):
                print('nop')
        'Change the PK of a related entity via foreign key cascade.\n\n        For databases that require "on update cascade", the mapper\n        has to identify the row by the new value, not the old, when\n        it does the update.\n\n        '
        (User, Address, users, addresses) = (self.classes.User, self.classes.Address, self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sess = fixture_session()
        (a1, a2) = (Address(username='ed', email='ed@host1'), Address(username='ed', email='ed@host2'))
        u1 = User(username='ed', addresses=[a1, a2])
        sess.add(u1)
        sess.flush()
        eq_(a1.username, 'ed')
        eq_(a2.username, 'ed')
        eq_(sess.execute(sa.select(addresses.c.username)).fetchall(), [('ed',), ('ed',)])
        u1.username = 'jack'
        a2.email = 'ed@host3'
        sess.flush()
        eq_(a1.username, 'jack')
        eq_(a2.username, 'jack')
        eq_(sess.execute(sa.select(addresses.c.username)).fetchall(), [('jack',), ('jack',)])

class JoinedInheritanceTest(fixtures.MappedTest):
    """Test cascades of pk->pk/fk on joined table inh."""
    __unsupported_on__ = ('mssql',)
    __requires__ = ('skip_mysql_on_windows',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            print('Hello World!')
        fk_args = _backend_specific_fk_args()
        Table('person', metadata, Column('name', String(50), primary_key=True), Column('type', String(50), nullable=False), test_needs_fk=True)
        Table('engineer', metadata, Column('name', String(50), ForeignKey('person.name', **fk_args), primary_key=True), Column('primary_language', String(50)), Column('boss_name', String(50), ForeignKey('manager.name', **fk_args)), test_needs_fk=True)
        Table('manager', metadata, Column('name', String(50), ForeignKey('person.name', **fk_args), primary_key=True), Column('paperwork', String(50)), test_needs_fk=True)
        Table('owner', metadata, Column('name', String(50), ForeignKey('manager.name', **fk_args), primary_key=True), Column('owner_name', String(50)), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            return 10

        class Person(cls.Comparable):
            pass

        class Engineer(Person):
            pass

        class Manager(Person):
            pass

        class Owner(Manager):
            pass

    def _mapping_fixture(self, threelevel, passive_updates):
        if False:
            for i in range(10):
                print('nop')
        (Person, Manager, Engineer, Owner) = self.classes('Person', 'Manager', 'Engineer', 'Owner')
        (person, manager, engineer, owner) = self.tables('person', 'manager', 'engineer', 'owner')
        self.mapper_registry.map_imperatively(Person, person, polymorphic_on=person.c.type, polymorphic_identity='person', passive_updates=passive_updates)
        self.mapper_registry.map_imperatively(Engineer, engineer, inherits=Person, polymorphic_identity='engineer', properties={'boss': relationship(Manager, primaryjoin=manager.c.name == engineer.c.boss_name, passive_updates=passive_updates)})
        self.mapper_registry.map_imperatively(Manager, manager, inherits=Person, polymorphic_identity='manager')
        if threelevel:
            self.mapper_registry.map_imperatively(Owner, owner, inherits=Manager, polymorphic_identity='owner')

    @testing.requires.on_update_cascade
    def test_pk_passive(self):
        if False:
            while True:
                i = 10
        self._test_pk(True)

    @testing.requires.non_updating_cascade
    def test_pk_nonpassive(self):
        if False:
            while True:
                i = 10
        self._test_pk(False)

    @testing.requires.on_update_cascade
    def test_fk_passive(self):
        if False:
            print('Hello World!')
        self._test_fk(True)

    @testing.requires.non_updating_cascade
    def test_fk_nonpassive(self):
        if False:
            print('Hello World!')
        self._test_fk(False)

    @testing.requires.on_update_cascade
    def test_pk_threelevel_passive(self):
        if False:
            print('Hello World!')
        self._test_pk_threelevel(True)

    @testing.requires.non_updating_cascade
    def test_pk_threelevel_nonpassive(self):
        if False:
            while True:
                i = 10
        self._test_pk_threelevel(False)

    @testing.requires.on_update_cascade
    def test_fk_threelevel_passive(self):
        if False:
            return 10
        self._test_fk_threelevel(True)

    @testing.requires.non_updating_cascade
    def test_fk_threelevel_nonpassive(self):
        if False:
            print('Hello World!')
        self._test_fk_threelevel(False)

    def _test_pk(self, passive_updates):
        if False:
            return 10
        (Engineer,) = self.classes('Engineer')
        self._mapping_fixture(False, passive_updates)
        sess = fixture_session()
        e1 = Engineer(name='dilbert', primary_language='java')
        sess.add(e1)
        sess.commit()
        e1.name = 'wally'
        e1.primary_language = 'c++'
        sess.commit()
        eq_(sess.execute(self.tables.engineer.select()).fetchall(), [('wally', 'c++', None)])
        eq_(e1.name, 'wally')
        e1.name = 'dogbert'
        sess.commit()
        eq_(e1.name, 'dogbert')
        eq_(sess.execute(self.tables.engineer.select()).fetchall(), [('dogbert', 'c++', None)])

    def _test_fk(self, passive_updates):
        if False:
            for i in range(10):
                print('nop')
        (Manager, Engineer) = self.classes('Manager', 'Engineer')
        self._mapping_fixture(False, passive_updates)
        sess = fixture_session()
        m1 = Manager(name='dogbert', paperwork='lots')
        (e1, e2) = (Engineer(name='dilbert', primary_language='java', boss=m1), Engineer(name='wally', primary_language='c++', boss=m1))
        sess.add_all([e1, e2, m1])
        sess.commit()
        eq_(e1.boss_name, 'dogbert')
        eq_(e2.boss_name, 'dogbert')
        eq_(sess.execute(self.tables.engineer.select().order_by(Engineer.name)).fetchall(), [('dilbert', 'java', 'dogbert'), ('wally', 'c++', 'dogbert')])
        sess.expire_all()
        m1.name = 'pointy haired'
        e1.primary_language = 'scala'
        e2.primary_language = 'cobol'
        sess.commit()
        eq_(e1.boss_name, 'pointy haired')
        eq_(e2.boss_name, 'pointy haired')
        eq_(sess.execute(self.tables.engineer.select().order_by(Engineer.name)).fetchall(), [('dilbert', 'scala', 'pointy haired'), ('wally', 'cobol', 'pointy haired')])

    def _test_pk_threelevel(self, passive_updates):
        if False:
            for i in range(10):
                print('nop')
        (Owner,) = self.classes('Owner')
        self._mapping_fixture(True, passive_updates)
        sess = fixture_session()
        o1 = Owner(name='dogbert', owner_name='dog')
        sess.add(o1)
        sess.commit()
        o1.name = 'pointy haired'
        o1.owner_name = 'pointy'
        sess.commit()
        eq_(sess.execute(self.tables.manager.select()).fetchall(), [('pointy haired', None)])
        eq_(sess.execute(self.tables.owner.select()).fetchall(), [('pointy haired', 'pointy')])
        eq_(o1.name, 'pointy haired')
        o1.name = 'catbert'
        sess.commit()
        eq_(o1.name, 'catbert')
        eq_(sess.execute(self.tables.manager.select()).fetchall(), [('catbert', None)])
        eq_(sess.execute(self.tables.owner.select()).fetchall(), [('catbert', 'pointy')])

    def _test_fk_threelevel(self, passive_updates):
        if False:
            i = 10
            return i + 15
        (Owner, Engineer) = self.classes('Owner', 'Engineer')
        self._mapping_fixture(True, passive_updates)
        sess = fixture_session()
        m1 = Owner(name='dogbert', paperwork='lots', owner_name='dog')
        (e1, e2) = (Engineer(name='dilbert', primary_language='java', boss=m1), Engineer(name='wally', primary_language='c++', boss=m1))
        sess.add_all([e1, e2, m1])
        sess.commit()
        eq_(e1.boss_name, 'dogbert')
        eq_(e2.boss_name, 'dogbert')
        sess.expire_all()
        m1.name = 'pointy haired'
        e1.primary_language = 'scala'
        e2.primary_language = 'cobol'
        sess.commit()
        eq_(e1.boss_name, 'pointy haired')
        eq_(e2.boss_name, 'pointy haired')
        eq_(sess.execute(self.tables.manager.select()).fetchall(), [('pointy haired', 'lots')])
        eq_(sess.execute(self.tables.owner.select()).fetchall(), [('pointy haired', 'dog')])

class UnsortablePKTest(fixtures.MappedTest):
    """Test integration with TypeEngine.sort_key_function"""

    class HashableDict(dict):

        def __hash__(self):
            if False:
                i = 10
                return i + 15
            return hash((self['x'], self['y']))

    @classmethod
    def define_tables(cls, metadata):
        if False:
            while True:
                i = 10

        class MyUnsortable(TypeDecorator):
            impl = String(10)
            cache_ok = True

            def process_bind_param(self, value, dialect):
                if False:
                    print('Hello World!')
                return '%s,%s' % (value['x'], value['y'])

            def process_result_value(self, value, dialect):
                if False:
                    print('Hello World!')
                rec = value.split(',')
                return cls.HashableDict({'x': rec[0], 'y': rec[1]})

            def sort_key_function(self, value):
                if False:
                    print('Hello World!')
                return (value['x'], value['y'])
        Table('data', metadata, Column('info', MyUnsortable(), primary_key=True), Column('int_value', Integer))

    @classmethod
    def setup_classes(cls):
        if False:
            print('Hello World!')

        class Data(cls.Comparable):
            pass

    @classmethod
    def setup_mappers(cls):
        if False:
            print('Hello World!')
        cls.mapper_registry.map_imperatively(cls.classes.Data, cls.tables.data)

    def test_updates_sorted(self):
        if False:
            for i in range(10):
                print('nop')
        Data = self.classes.Data
        s = fixture_session()
        s.add_all([Data(info=self.HashableDict(x='a', y='b')), Data(info=self.HashableDict(x='a', y='a')), Data(info=self.HashableDict(x='b', y='b')), Data(info=self.HashableDict(x='b', y='a'))])
        s.commit()
        (aa, ab, ba, bb) = s.query(Data).order_by(Data.info).all()
        counter = itertools.count()
        ab.int_value = bindparam(key=None, callable_=lambda : next(counter))
        ba.int_value = bindparam(key=None, callable_=lambda : next(counter))
        bb.int_value = bindparam(key=None, callable_=lambda : next(counter))
        aa.int_value = bindparam(key=None, callable_=lambda : next(counter))
        s.commit()
        eq_(s.query(Data.int_value).order_by(Data.info).all(), [(0,), (1,), (2,), (3,)])

class JoinedInheritancePKOnFKTest(fixtures.MappedTest):
    """Test cascades of pk->non-pk/fk on joined table inh."""
    __unsupported_on__ = ('mssql',)
    __requires__ = ('skip_mysql_on_windows',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        if False:
            i = 10
            return i + 15
        fk_args = _backend_specific_fk_args()
        Table('person', metadata, Column('name', String(50), primary_key=True), Column('type', String(50), nullable=False), test_needs_fk=True)
        Table('engineer', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('person_name', String(50), ForeignKey('person.name', **fk_args)), Column('primary_language', String(50)), test_needs_fk=True)

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')

        class Person(cls.Comparable):
            pass

        class Engineer(Person):
            pass

    def _test_pk(self, passive_updates):
        if False:
            i = 10
            return i + 15
        (Person, person, Engineer, engineer) = (self.classes.Person, self.tables.person, self.classes.Engineer, self.tables.engineer)
        self.mapper_registry.map_imperatively(Person, person, polymorphic_on=person.c.type, polymorphic_identity='person', passive_updates=passive_updates)
        self.mapper_registry.map_imperatively(Engineer, engineer, inherits=Person, polymorphic_identity='engineer')
        sess = fixture_session()
        e1 = Engineer(name='dilbert', primary_language='java')
        sess.add(e1)
        sess.commit()
        e1.name = 'wally'
        e1.primary_language = 'c++'
        sess.flush()
        eq_(e1.person_name, 'wally')
        sess.expire_all()
        eq_(e1.primary_language, 'c++')

    @testing.requires.on_update_cascade
    def test_pk_passive(self):
        if False:
            while True:
                i = 10
        self._test_pk(True)

    def test_pk_nonpassive(self):
        if False:
            while True:
                i = 10
        self._test_pk(False)
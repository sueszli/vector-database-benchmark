"""basic tests of lazy loaded attributes"""
from sqlalchemy import select
from sqlalchemy import testing
from sqlalchemy.orm import immediateload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.testing import eq_
from sqlalchemy.testing import is_
from sqlalchemy.testing.fixtures import fixture_session
from test.orm import _fixtures

class ImmediateTest(_fixtures.FixtureTest):
    run_inserts = 'once'
    run_deletes = None

    @testing.combinations(('raise',), ('raise_on_sql',), ('select',), 'immediate')
    def test_basic_option(self, default_lazy):
        if False:
            return 10
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, lazy=default_lazy)})
        sess = fixture_session()
        result = sess.query(User).options(immediateload(User.addresses)).filter(users.c.id == 7).all()
        eq_(len(sess.identity_map), 2)
        sess.close()
        eq_([User(id=7, addresses=[Address(id=1, email_address='jack@bean.com')])], result)

    @testing.combinations(('raise',), ('raise_on_sql',), ('select',), 'immediate')
    def test_basic_option_m2o(self, default_lazy):
        if False:
            i = 10
            return i + 15
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, lazy=default_lazy)})
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        result = sess.query(Address).options(immediateload(Address.user)).filter(Address.id == 1).all()
        eq_(len(sess.identity_map), 2)
        sess.close()
        eq_([Address(id=1, email_address='jack@bean.com', user=User(id=7))], result)

    def test_basic(self):
        if False:
            print('Hello World!')
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, lazy='immediate')})
        sess = fixture_session()
        result = sess.query(User).filter(users.c.id == 7).all()
        eq_(len(sess.identity_map), 2)
        sess.close()
        eq_([User(id=7, addresses=[Address(id=1, email_address='jack@bean.com')])], result)

    @testing.combinations(('joined',), ('selectin',), ('subquery',))
    def test_m2one_side(self, o2m_lazy):
        if False:
            while True:
                i = 10
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, lazy='immediate', back_populates='addresses')})
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, lazy=o2m_lazy, back_populates='user')})
        sess = fixture_session()
        u1 = sess.query(User).filter(users.c.id == 7).one()
        sess.close()
        assert 'addresses' in u1.__dict__
        assert 'user' in u1.addresses[0].__dict__

    @testing.combinations(('immediate',), ('joined',), ('selectin',), ('subquery',))
    def test_o2mone_side(self, m2o_lazy):
        if False:
            i = 10
            return i + 15
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User, lazy=m2o_lazy, back_populates='addresses')})
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, lazy='immediate', back_populates='user')})
        sess = fixture_session()
        u1 = sess.query(User).filter(users.c.id == 7).one()
        sess.close()
        assert 'addresses' in u1.__dict__
        assert 'user' not in u1.addresses[0].__dict__

class SelfReferentialTest(_fixtures.FixtureTest):
    run_inserts = None
    run_deletes = 'each'

    @testing.fixture
    def node_fixture(self):
        if False:
            while True:
                i = 10
        Node = self.classes.Node
        nodes = self.tables.nodes

        def go(join_depth):
            if False:
                return 10
            self.mapper_registry.map_imperatively(Node, nodes, properties={'parent': relationship(Node, remote_side=nodes.c.id, lazy='immediate', join_depth=join_depth)})
            return Node
        yield go
        with Session(testing.db) as sess:
            for node in sess.scalars(select(Node)):
                sess.delete(node)
            sess.commit()

    @testing.variation('persistence', ['expunge', 'keep', 'reload'])
    @testing.combinations((None,), (1,), (2,), argnames='join_depth')
    def test_self_referential_recursive(self, persistence, join_depth, node_fixture):
        if False:
            for i in range(10):
                print('nop')
        'test #10139'
        Node = node_fixture(join_depth)
        sess = fixture_session()
        n0 = Node(data='n0')
        n1 = Node(data='n1')
        n2 = Node(data='n2')
        n1.parent = n0
        n2.parent = n1
        sess.add_all([n0, n1, n2])
        sess.commit()
        if persistence.expunge or persistence.reload:
            sess.close()
        if persistence.reload:
            sess.add(n1)
            sess.add(n0)
        n2 = sess.query(Node).filter(Node.data == 'n2').one()
        if persistence.expunge and (join_depth is None or join_depth < 1):
            expected_count = 1
        else:
            expected_count = 0
        with self.assert_statement_count(testing.db, expected_count):
            if persistence.keep or persistence.reload:
                is_(n2.parent, n1)
            else:
                eq_(n2.parent, Node(data='n1'))
            n1 = n2.parent
        n1.parent_id
        if persistence.expunge and (join_depth is None or join_depth < 2):
            expected_count = 1
        else:
            expected_count = 0
        with self.assert_statement_count(testing.db, expected_count):
            if persistence.keep or persistence.reload:
                is_(n1.parent, n0)
            else:
                eq_(n1.parent, Node(data='n0'))
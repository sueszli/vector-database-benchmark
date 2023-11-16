import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import literal_column
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import testing
from sqlalchemy.orm import immediateload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import expect_warnings
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table
from test.orm import _fixtures

class NonRecursiveTest(_fixtures.FixtureTest):

    @classmethod
    def setup_mappers(cls):
        if False:
            return 10
        cls._setup_stock_mapping()

    @testing.combinations(selectinload, immediateload, argnames='loader')
    def test_no_recursion_depth_non_self_referential(self, loader):
        if False:
            i = 10
            return i + 15
        User = self.classes.User
        sess = fixture_session()
        stmt = select(User).options(selectinload(User.addresses, recursion_depth=-1))
        with expect_raises_message(sa.exc.InvalidRequestError, 'recursion_depth option on relationship User.addresses not valid'):
            sess.execute(stmt).all()

class _NodeTest:

    @classmethod
    def define_tables(cls, metadata):
        if False:
            for i in range(10):
                print('nop')
        Table('nodes', metadata, Column('id', Integer, primary_key=True), Column('parent_id', Integer, ForeignKey('nodes.id')), Column('data', String(30)))

    @classmethod
    def setup_mappers(cls):
        if False:
            while True:
                i = 10
        nodes = cls.tables.nodes
        Node = cls.classes.Node
        cls.mapper_registry.map_imperatively(Node, nodes, properties={'children': relationship(Node)})

    @classmethod
    def setup_classes(cls):
        if False:
            print('Hello World!')

        class Node(cls.Comparable):

            def append(self, node):
                if False:
                    while True:
                        i = 10
                self.children.append(node)

class ShallowRecursiveTest(_NodeTest, fixtures.MappedTest):

    @classmethod
    def insert_data(cls, connection):
        if False:
            return 10
        Node = cls.classes.Node
        n1 = Node(data='n1')
        n1.append(Node(data='n11'))
        n1.append(Node(data='n12'))
        n1.append(Node(data='n13'))
        n1.children[0].children = [Node(data='n111'), Node(data='n112')]
        n1.children[1].append(Node(data='n121'))
        n1.children[1].append(Node(data='n122'))
        n1.children[1].append(Node(data='n123'))
        n2 = Node(data='n2')
        n2.append(Node(data='n21'))
        n2.children[0].append(Node(data='n211'))
        n2.children[0].append(Node(data='n212'))
        with Session(connection) as sess:
            sess.add(n1)
            sess.add(n2)
            sess.commit()

    @testing.fixture
    def data_fixture(self):
        if False:
            for i in range(10):
                print('nop')
        Node = self.classes.Node

        def go(sess):
            if False:
                i = 10
                return i + 15
            (n1, n2) = sess.scalars(select(Node).where(Node.data.in_(['n1', 'n2'])).order_by(Node.id)).all()
            return (n1, n2)
        return go

    def _full_structure(self):
        if False:
            while True:
                i = 10
        Node = self.classes.Node
        return [Node(data='n1', children=[Node(data='n11'), Node(data='n12', children=[Node(data='n121'), Node(data='n122'), Node(data='n123')]), Node(data='n13')]), Node(data='n2', children=[Node(data='n21', children=[Node(data='n211'), Node(data='n212')])])]

    @testing.combinations((selectinload, 4), (immediateload, 14), argnames='loader,expected_sql_count')
    def test_recursion_depth_opt(self, data_fixture, loader, expected_sql_count):
        if False:
            return 10
        Node = self.classes.Node
        sess = fixture_session()
        (n1, n2) = data_fixture(sess)

        def go():
            if False:
                i = 10
                return i + 15
            return sess.query(Node).filter(Node.data.in_(['n1', 'n2'])).options(loader(Node.children, recursion_depth=-1)).order_by(Node.data).all()
        result = self.assert_sql_count(testing.db, go, expected_sql_count)
        sess.close()
        eq_(result, self._full_structure())

class DeepRecursiveTest(_NodeTest, fixtures.MappedTest):

    @classmethod
    def insert_data(cls, connection):
        if False:
            return 10
        nodes = cls.tables.nodes
        connection.execute(nodes.insert(), [{'id': i, 'parent_id': i - 1 if i > 1 else None} for i in range(1, 201)])
        connection.commit()

    @testing.fixture
    def limited_cache_conn(self, connection):
        if False:
            while True:
                i = 10
        connection.engine._compiled_cache.clear()
        assert_limit = 0

        def go(limit):
            if False:
                return 10
            nonlocal assert_limit
            assert_limit = limit
            return connection
        yield go
        clen = len(connection.engine._compiled_cache)
        assert clen > 1
        assert clen < assert_limit, f'cache grew to {clen}'

    def _stack_loaders(self, loader_fn, depth):
        if False:
            i = 10
            return i + 15
        Node = self.classes.Node
        opt = loader_fn(Node.children)
        while depth:
            opt = getattr(opt, loader_fn.__name__)(Node.children)
            depth -= 1
        return opt

    def _assert_depth(self, obj, depth):
        if False:
            while True:
                i = 10
        stack = [obj]
        depth += 1
        while stack and depth:
            n = stack.pop(0)
            stack.extend(n.__dict__['children'])
            depth -= 1
        for n in stack:
            assert 'children' not in n.__dict__

    @testing.combinations(selectinload, immediateload, argnames='loader_fn')
    @testing.combinations(1, 15, 25, 185, 78, argnames='depth')
    def test_recursion_depth(self, loader_fn, depth, limited_cache_conn):
        if False:
            print('Hello World!')
        connection = limited_cache_conn(6)
        Node = self.classes.Node
        for i in range(2):
            stmt = select(Node).filter(Node.id == 1).options(loader_fn(Node.children, recursion_depth=depth))
            with Session(connection) as s:
                result = s.scalars(stmt)
                self._assert_depth(result.one(), depth)

    @testing.combinations(selectinload, immediateload, argnames='loader_fn')
    def test_unlimited_recursion(self, loader_fn, limited_cache_conn):
        if False:
            return 10
        connection = limited_cache_conn(6)
        Node = self.classes.Node
        for i in range(2):
            stmt = select(Node).filter(Node.id == 1).options(loader_fn(Node.children, recursion_depth=-1))
            with Session(connection) as s:
                result = s.scalars(stmt)
                self._assert_depth(result.one(), 200)

    @testing.combinations(selectinload, immediateload, argnames='loader_fn')
    @testing.combinations(4, 9, 12, 25, 41, 55, argnames='depth')
    @testing.variation('disable_cache', [True, False])
    def test_warning_w_no_recursive_opt(self, loader_fn, depth, limited_cache_conn, disable_cache):
        if False:
            i = 10
            return i + 15
        connection = limited_cache_conn(27)
        Node = self.classes.Node
        for i in range(2):
            stmt = select(Node).filter(Node.id == 1).options(self._stack_loaders(loader_fn, depth))
            if disable_cache:
                exec_opts = dict(compiled_cache=None)
            else:
                exec_opts = {}
            if depth > 8 and (not disable_cache):
                with expect_warnings('Loader depth for query is excessively deep; caching will be disabled for additional loaders.'):
                    with Session(connection) as s:
                        result = s.scalars(stmt, execution_options=exec_opts)
                        self._assert_depth(result.one(), depth)
            else:
                with Session(connection) as s:
                    result = s.scalars(stmt, execution_options=exec_opts)
                    self._assert_depth(result.one(), depth)
        if disable_cache:
            clen = len(connection.engine._compiled_cache)
            assert clen == 0
            connection.execute(select(1))
            connection.execute(select(1).where(literal_column('1') == 1))
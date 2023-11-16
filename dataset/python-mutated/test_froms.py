from sqlalchemy import and_
from sqlalchemy import asc
from sqlalchemy import desc
from sqlalchemy import exc as sa_exc
from sqlalchemy import exists
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import literal
from sqlalchemy import literal_column
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import true
from sqlalchemy import union
from sqlalchemy import util
from sqlalchemy.engine import default
from sqlalchemy.orm import aliased
from sqlalchemy.orm import backref
from sqlalchemy.orm import clear_mappers
from sqlalchemy.orm import column_property
from sqlalchemy.orm import configure_mappers
from sqlalchemy.orm import contains_eager
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.orm.context import ORMSelectCompileState
from sqlalchemy.sql import column
from sqlalchemy.sql import table
from sqlalchemy.sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from sqlalchemy.testing import assert_raises
from sqlalchemy.testing import assert_raises_message
from sqlalchemy.testing import AssertsCompiledSQL
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import in_
from sqlalchemy.testing import is_
from sqlalchemy.testing.entities import ComparableEntity
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from test.orm import _fixtures

class QueryTest(_fixtures.FixtureTest):
    run_setup_mappers = 'once'
    run_inserts = 'once'
    run_deletes = None

    @classmethod
    def setup_mappers(cls):
        if False:
            return 10
        (Node, composite_pk_table, users, Keyword, items, Dingaling, order_items, item_keywords, Item, User, dingalings, Address, keywords, CompositePk, nodes, Order, orders, addresses) = (cls.classes.Node, cls.tables.composite_pk_table, cls.tables.users, cls.classes.Keyword, cls.tables.items, cls.classes.Dingaling, cls.tables.order_items, cls.tables.item_keywords, cls.classes.Item, cls.classes.User, cls.tables.dingalings, cls.classes.Address, cls.tables.keywords, cls.classes.CompositePk, cls.tables.nodes, cls.classes.Order, cls.tables.orders, cls.tables.addresses)
        cls.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, backref='user', order_by=addresses.c.id), 'orders': relationship(Order, backref='user', order_by=orders.c.id)})
        cls.mapper_registry.map_imperatively(Address, addresses, properties={'dingaling': relationship(Dingaling, uselist=False, backref='address')})
        cls.mapper_registry.map_imperatively(Dingaling, dingalings)
        cls.mapper_registry.map_imperatively(Order, orders, properties={'items': relationship(Item, secondary=order_items, order_by=items.c.id), 'address': relationship(Address)})
        cls.mapper_registry.map_imperatively(Item, items, properties={'keywords': relationship(Keyword, secondary=item_keywords)})
        cls.mapper_registry.map_imperatively(Keyword, keywords)
        cls.mapper_registry.map_imperatively(Node, nodes, properties={'children': relationship(Node, backref=backref('parent', remote_side=[nodes.c.id]))})
        cls.mapper_registry.map_imperatively(CompositePk, composite_pk_table)
        configure_mappers()

class QueryCorrelatesLikeSelect(QueryTest, AssertsCompiledSQL):
    __dialect__ = 'default'
    query_correlated = 'SELECT users.name AS users_name, (SELECT count(addresses.id) AS count_1 FROM addresses WHERE addresses.user_id = users.id) AS anon_1 FROM users'
    query_not_correlated = 'SELECT users.name AS users_name, (SELECT count(addresses.id) AS count_1 FROM addresses, users WHERE addresses.user_id = users.id) AS anon_1 FROM users'

    def test_scalar_subquery_select_auto_correlate(self):
        if False:
            for i in range(10):
                print('nop')
        (addresses, users) = (self.tables.addresses, self.tables.users)
        query = select(func.count(addresses.c.id)).where(addresses.c.user_id == users.c.id).scalar_subquery()
        query = select(users.c.name.label('users_name'), query)
        self.assert_compile(query, self.query_correlated, dialect=default.DefaultDialect())

    def test_scalar_subquery_select_explicit_correlate(self):
        if False:
            return 10
        (addresses, users) = (self.tables.addresses, self.tables.users)
        query = select(func.count(addresses.c.id)).where(addresses.c.user_id == users.c.id).correlate(users).scalar_subquery()
        query = select(users.c.name.label('users_name'), query)
        self.assert_compile(query, self.query_correlated, dialect=default.DefaultDialect())

    def test_scalar_subquery_select_correlate_off(self):
        if False:
            while True:
                i = 10
        (addresses, users) = (self.tables.addresses, self.tables.users)
        query = select(func.count(addresses.c.id)).where(addresses.c.user_id == users.c.id).correlate(None).scalar_subquery()
        query = select(users.c.name.label('users_name'), query)
        self.assert_compile(query, self.query_not_correlated, dialect=default.DefaultDialect())

    def test_scalar_subquery_query_auto_correlate(self):
        if False:
            return 10
        sess = fixture_session()
        (Address, User) = (self.classes.Address, self.classes.User)
        query = sess.query(func.count(Address.id)).filter(Address.user_id == User.id).scalar_subquery()
        query = sess.query(User.name, query)
        self.assert_compile(query, self.query_correlated, dialect=default.DefaultDialect())

    def test_scalar_subquery_query_explicit_correlate(self):
        if False:
            return 10
        sess = fixture_session()
        (Address, User) = (self.classes.Address, self.classes.User)
        query = sess.query(func.count(Address.id)).filter(Address.user_id == User.id).correlate(self.tables.users).scalar_subquery()
        query = sess.query(User.name, query)
        self.assert_compile(query, self.query_correlated, dialect=default.DefaultDialect())

    @testing.combinations(False, None)
    def test_scalar_subquery_query_correlate_off(self, value):
        if False:
            i = 10
            return i + 15
        sess = fixture_session()
        (Address, User) = (self.classes.Address, self.classes.User)
        query = sess.query(func.count(Address.id)).filter(Address.user_id == User.id).correlate(value).scalar_subquery()
        query = sess.query(User.name, query)
        self.assert_compile(query, self.query_not_correlated, dialect=default.DefaultDialect())

    def test_correlate_to_union(self):
        if False:
            return 10
        User = self.classes.User
        sess = fixture_session()
        q = sess.query(User)
        q = sess.query(User).union(q)
        u_alias = aliased(User)
        raw_subq = exists().where(u_alias.id > User.id)
        orm_subq = sess.query(u_alias).filter(u_alias.id > User.id).exists()
        self.assert_compile(q.add_columns(raw_subq), 'SELECT anon_1.users_id AS anon_1_users_id, anon_1.users_name AS anon_1_users_name, EXISTS (SELECT * FROM users AS users_1 WHERE users_1.id > anon_1.users_id) AS anon_2 FROM (SELECT users.id AS users_id, users.name AS users_name FROM users UNION SELECT users.id AS users_id, users.name AS users_name FROM users) AS anon_1')
        self.assert_compile(q.add_columns(orm_subq), 'SELECT anon_1.users_id AS anon_1_users_id, anon_1.users_name AS anon_1_users_name, EXISTS (SELECT 1 FROM users AS users_1 WHERE users_1.id > anon_1.users_id) AS anon_2 FROM (SELECT users.id AS users_id, users.name AS users_name FROM users UNION SELECT users.id AS users_id, users.name AS users_name FROM users) AS anon_1')

    def test_correlate_to_union_w_labels_newstyle(self):
        if False:
            return 10
        User = self.classes.User
        q = select(User).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        q = select(User).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).union(q).subquery()
        u_alias = aliased(User)
        raw_subq = exists().where(u_alias.id > q.c[0])
        self.assert_compile(select(q, raw_subq).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL), 'SELECT anon_1.users_id AS anon_1_users_id, anon_1.users_name AS anon_1_users_name, EXISTS (SELECT * FROM users AS users_1 WHERE users_1.id > anon_1.users_id) AS anon_2 FROM (SELECT users.id AS users_id, users.name AS users_name FROM users UNION SELECT users.id AS users_id, users.name AS users_name FROM users) AS anon_1')

    def test_correlate_to_union_newstyle(self):
        if False:
            print('Hello World!')
        User = self.classes.User
        q = select(User)
        q = select(User).union(q).subquery()
        u_alias = aliased(User)
        raw_subq = exists().where(u_alias.id > q.c[0])
        self.assert_compile(select(q, raw_subq), 'SELECT anon_1.id, anon_1.name, EXISTS (SELECT * FROM users AS users_1 WHERE users_1.id > anon_1.id) AS anon_2 FROM (SELECT users.id AS id, users.name AS name FROM users UNION SELECT users.id AS id, users.name AS name FROM users) AS anon_1')

class RawSelectTest(QueryTest, AssertsCompiledSQL):
    """compare a bunch of select() tests with the equivalent Query using
    straight table/columns.

    Results should be the same as Query should act as a select() pass-
    thru for ClauseElement entities.

    """
    __dialect__ = 'default'

    def test_select(self):
        if False:
            i = 10
            return i + 15
        (addresses, users) = (self.tables.addresses, self.tables.users)
        sess = fixture_session()
        self.assert_compile(sess.query(users).select_from(users.select().subquery()).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).statement, 'SELECT users.id AS users_id, users.name AS users_name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1, users')
        self.assert_compile(sess.query(users, exists(text('1')).select_from(addresses)).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).statement, 'SELECT users.id AS users_id, users.name AS users_name, EXISTS (SELECT 1 FROM addresses) AS anon_1 FROM users')
        s = sess.query(addresses.c.id.label('id'), addresses.c.email_address.label('email')).filter(addresses.c.user_id == users.c.id).correlate(users).statement.alias()
        self.assert_compile(sess.query(users, s.c.email).select_from(users.join(s, s.c.id == users.c.id)).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).statement, 'SELECT users.id AS users_id, users.name AS users_name, anon_1.email AS anon_1_email FROM users JOIN (SELECT addresses.id AS id, addresses.email_address AS email FROM addresses, users WHERE addresses.user_id = users.id) AS anon_1 ON anon_1.id = users.id')
        x = func.lala(users.c.id).label('foo')
        self.assert_compile(sess.query(x).filter(x == 5).statement, 'SELECT lala(users.id) AS foo FROM users WHERE lala(users.id) = :param_1')
        self.assert_compile(sess.query(func.sum(x).label('bar')).statement, 'SELECT sum(lala(users.id)) AS bar FROM users')

class EntityFromSubqueryTest(QueryTest, AssertsCompiledSQL):
    __dialect__ = 'default'

    def test_filter(self):
        if False:
            print('Hello World!')
        User = self.classes.User
        subq = select(User).filter(User.id.in_([8, 9])).subquery()
        q = fixture_session().query(aliased(User, subq))
        eq_([User(id=8), User(id=9)], q.all())
        subq = select(User).order_by(User.id).slice(1, 3).subquery()
        q = fixture_session().query(aliased(User, subq))
        eq_([User(id=8), User(id=9)], q.all())
        subq = select(User).filter(User.id.in_([8, 9])).subquery()
        u = aliased(User, subq)
        q = fixture_session().query(u).order_by(u.id)
        eq_([User(id=8)], list(q[0:1]))

    def test_join(self):
        if False:
            for i in range(10):
                print('nop')
        (User, Address) = (self.classes.User, self.classes.Address)
        stmt = select(User).filter(User.id.in_([8, 9])).subquery()
        u = aliased(User, stmt)
        q = fixture_session().query(u).join(u.addresses).add_entity(Address).order_by(u.id, Address.id)
        eq_([(User(id=8), Address(id=2)), (User(id=8), Address(id=3)), (User(id=8), Address(id=4)), (User(id=9), Address(id=5))], q.all())

    def test_group_by(self):
        if False:
            print('Hello World!')
        Address = self.classes.Address
        subq = select(Address.user_id, func.count(Address.id).label('count')).group_by(Address.user_id).order_by(Address.user_id).subquery()
        aq = aliased(Address, subq)
        q = fixture_session().query(aq.user_id, subq.c.count)
        eq_(q.all(), [(7, 1), (8, 3), (9, 1)])
        subq = select(Address.user_id, Address.id)
        aq = aliased(Address, subq)
        q = fixture_session().query(aq.user_id, func.count(aq.id)).group_by(aq.user_id).order_by(aq.user_id)
        eq_(q.all(), [(7, 1), (8, 3), (9, 1)])

    def test_error_w_aliased_against_select(self):
        if False:
            return 10
        User = self.classes.User
        s = fixture_session()
        stmt = select(User.id)
        assert_raises_message(sa_exc.ArgumentError, "Column expression, FROM clause, or other .* expected, got <sqlalchemy.sql.selectable.Select .*> object resolved from <AliasedClass .* User> object. To create a FROM clause from a <class 'sqlalchemy.sql.selectable.Select'> object", s.query, aliased(User, stmt))

    def test_having(self):
        if False:
            i = 10
            return i + 15
        User = self.classes.User
        s = fixture_session()
        stmt = select(User.id).group_by(User.id).having(User.id > 5).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        q = s.query(aliased(User, stmt))
        self.assert_compile(q, 'SELECT anon_1.users_id AS anon_1_users_id FROM (SELECT users.id AS users_id FROM users GROUP BY users.id HAVING users.id > :id_1) AS anon_1')

    def test_no_joinedload(self):
        if False:
            for i in range(10):
                print('nop')
        User = self.classes.User
        s = fixture_session()
        subq = select(User).options(joinedload(User.addresses)).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        uq = aliased(User, subq)
        q = s.query(uq)
        self.assert_compile(q.statement, 'SELECT anon_1.users_id, anon_1.users_name FROM (SELECT users.id AS users_id, users.name AS users_name FROM users) AS anon_1')
        self.assert_compile(q.options(joinedload(uq.addresses)).statement, 'SELECT anon_1.users_id, anon_1.users_name, addresses_1.id, addresses_1.user_id, addresses_1.email_address FROM (SELECT users.id AS users_id, users.name AS users_name FROM users) AS anon_1 LEFT OUTER JOIN addresses AS addresses_1 ON anon_1.users_id = addresses_1.user_id ORDER BY addresses_1.id')

    def test_aliases(self):
        if False:
            return 10
        'test that aliased objects are accessible externally to a from_self()\n        call.'
        (User, Address) = (self.classes.User, self.classes.Address)
        s = fixture_session()
        ualias = aliased(User)
        subq = select(User, ualias).filter(User.id > ualias.id).subquery()
        uq1 = aliased(User, subq)
        uq2 = aliased(ualias, subq)
        q = s.query(uq1.name, uq2.name).order_by(uq1.name, uq2.name)
        self.assert_compile(q, 'SELECT anon_1.name AS anon_1_name, anon_1.name_1 AS anon_1_name_1 FROM (SELECT users.id AS id, users.name AS name, users_1.id AS id_1, users_1.name AS name_1 FROM users, users AS users_1 WHERE users.id > users_1.id) AS anon_1 ORDER BY anon_1.name, anon_1.name_1')
        eq_(q.all(), [('chuck', 'ed'), ('chuck', 'fred'), ('chuck', 'jack'), ('ed', 'jack'), ('fred', 'ed'), ('fred', 'jack')])
        q = s.query(uq1.name, uq2.name).filter(uq2.name == 'ed').order_by(uq1.name, uq2.name)
        eq_(q.all(), [('chuck', 'ed'), ('fred', 'ed')])
        q = s.query(uq2.name, Address.email_address).join(uq2.addresses).order_by(uq2.name, Address.email_address)
        eq_(q.all(), [('ed', 'fred@fred.com'), ('jack', 'ed@bettyboop.com'), ('jack', 'ed@lala.com'), ('jack', 'ed@wood.com'), ('jack', 'fred@fred.com')])

    def test_nested_aliases_none_to_none(self):
        if False:
            return 10
        'test #7576'
        User = self.classes.User
        u1 = aliased(User)
        u2 = aliased(u1)
        self.assert_compile(select(u2), 'SELECT users_1.id, users_1.name FROM users AS users_1')

    def test_nested_alias_none_to_subquery(self):
        if False:
            i = 10
            return i + 15
        'test #7576'
        User = self.classes.User
        subq = select(User.id, User.name).subquery()
        u1 = aliased(User, subq)
        self.assert_compile(select(u1), 'SELECT anon_1.id, anon_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1')
        u2 = aliased(u1)
        self.assert_compile(select(u2), 'SELECT anon_1.id, anon_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1')

    def test_nested_alias_subquery_to_subquery_w_replace(self):
        if False:
            while True:
                i = 10
        'test #7576'
        User = self.classes.User
        subq = select(User.id, User.name).subquery()
        u1 = aliased(User, subq)
        self.assert_compile(select(u1), 'SELECT anon_1.id, anon_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1')
        u2 = aliased(u1, subq)
        self.assert_compile(select(u2), 'SELECT anon_1.id, anon_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1')

    def test_nested_alias_subquery_to_subquery_w_adaption(self):
        if False:
            print('Hello World!')
        'test #7576'
        User = self.classes.User
        inner_subq = select(User.id, User.name).subquery()
        u1 = aliased(User, inner_subq)
        self.assert_compile(select(u1), 'SELECT anon_1.id, anon_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1')
        outer_subq = select(u1.id, u1.name).subquery()
        u2 = aliased(u1, outer_subq)
        self.assert_compile(select(u2), 'SELECT anon_1.id, anon_1.name FROM (SELECT anon_2.id AS id, anon_2.name AS name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_2) AS anon_1')
        outer_subq = select(u1.id, u1.name, User.id, User.name).where(u1.id > User.id).subquery()
        u2 = aliased(u1, outer_subq)
        self.assert_compile(select(u2), 'SELECT anon_1.id, anon_1.name FROM (SELECT anon_2.id AS id, anon_2.name AS name, users.id AS id_1, users.name AS name_1 FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_2, users WHERE anon_2.id > users.id) AS anon_1')

    def test_nested_alias_subquery_w_alias_to_none(self):
        if False:
            i = 10
            return i + 15
        'test #7576'
        User = self.classes.User
        u1 = aliased(User)
        self.assert_compile(select(u1), 'SELECT users_1.id, users_1.name FROM users AS users_1')
        subq = select(User.id, User.name, u1.id, u1.name).where(User.id > u1.id).subquery()
        u2 = aliased(u1, subq)
        self.assert_compile(select(u2), 'SELECT anon_1.id_1, anon_1.name_1 FROM (SELECT users.id AS id, users.name AS name, users_1.id AS id_1, users_1.name AS name_1 FROM users, users AS users_1 WHERE users.id > users_1.id) AS anon_1')
        subq = select(User.id, User.name).subquery()
        u2 = aliased(u1, subq)
        self.assert_compile(select(u2), 'SELECT users_1.id, users_1.name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1, users AS users_1')

    def test_multiple_entities(self):
        if False:
            i = 10
            return i + 15
        (User, Address) = (self.classes.User, self.classes.Address)
        sess = fixture_session()
        subq = select(User, Address).filter(User.id == Address.user_id).filter(Address.id.in_([2, 5])).subquery()
        uq = aliased(User, subq)
        aq = aliased(Address, subq)
        eq_(sess.query(uq, aq).all(), [(User(id=8), Address(id=2)), (User(id=9), Address(id=5))])
        eq_(sess.query(uq, aq).options(joinedload(uq.addresses)).first(), (User(id=8, addresses=[Address(), Address(), Address()]), Address(id=2)))

    def test_multiple_with_column_entities_oldstyle(self):
        if False:
            while True:
                i = 10
        User = self.classes.User
        subq = select(User.id).subquery()
        uq = aliased(User, subq)
        subq2 = select(uq.id).add_columns(func.count().label('foo')).group_by(uq.id).order_by(uq.id).subquery()
        uq2 = aliased(User, subq2)
        sess = fixture_session()
        eq_(sess.query(uq2.id, subq2.c.foo).all(), [(7, 1), (8, 1), (9, 1), (10, 1)])

    def test_multiple_with_column_entities_newstyle(self):
        if False:
            return 10
        User = self.classes.User
        sess = fixture_session()
        q1 = sess.query(User.id)
        subq1 = aliased(User, q1.subquery())
        q2 = sess.query(subq1.id).add_columns(func.count().label('foo'))
        q2 = q2.group_by(subq1.id).order_by(subq1.id).subquery()
        q3 = sess.query(q2)
        eq_(q3.all(), [(7, 1), (8, 1), (9, 1), (10, 1)])
        q3 = select(q2)
        eq_(sess.execute(q3).fetchall(), [(7, 1), (8, 1), (9, 1), (10, 1)])

class ColumnAccessTest(QueryTest, AssertsCompiledSQL):
    """test access of columns after _from_selectable has been applied"""
    __dialect__ = 'default'

    def test_select_from_no_aliasing(self):
        if False:
            while True:
                i = 10
        User = self.classes.User
        sess = fixture_session()
        q = sess.query(User)
        q = sess.query(User).select_from(q.statement.subquery())
        self.assert_compile(q.filter(User.name == 'ed'), 'SELECT users.id AS users_id, users.name AS users_name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1, users WHERE users.name = :name_1')

    def test_anonymous_expression_oldstyle(self):
        if False:
            return 10
        from sqlalchemy.sql import column
        sess = fixture_session()
        (c1, c2) = (column('c1'), column('c2'))
        q1 = sess.query(c1, c2).filter(c1 == 'dog')
        q2 = sess.query(c1, c2).filter(c1 == 'cat')
        q3 = q1.union(q2)
        self.assert_compile(q3.order_by(c1), 'SELECT anon_1.c1 AS anon_1_c1, anon_1.c2 AS anon_1_c2 FROM (SELECT c1, c2 WHERE c1 = :c1_1 UNION SELECT c1, c2 WHERE c1 = :c1_2) AS anon_1 ORDER BY anon_1.c1')

    def test_anonymous_expression_newstyle(self):
        if False:
            for i in range(10):
                print('nop')
        from sqlalchemy.sql import column
        (c1, c2) = (column('c1'), column('c2'))
        q1 = select(c1, c2).where(c1 == 'dog')
        q2 = select(c1, c2).where(c1 == 'cat')
        subq = q1.union(q2).subquery()
        q3 = select(subq).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        self.assert_compile(q3.order_by(subq.c.c1), 'SELECT anon_1.c1 AS anon_1_c1, anon_1.c2 AS anon_1_c2 FROM (SELECT c1, c2 WHERE c1 = :c1_1 UNION SELECT c1, c2 WHERE c1 = :c1_2) AS anon_1 ORDER BY anon_1.c1')

    def test_table_anonymous_expression_from_self_twice_newstyle(self):
        if False:
            while True:
                i = 10
        from sqlalchemy.sql import column
        t1 = table('t1', column('c1'), column('c2'))
        stmt = select(t1.c.c1, t1.c.c2).where(t1.c.c1 == 'dog').set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        subq1 = stmt.subquery('anon_2').select().set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        subq2 = subq1.subquery('anon_1')
        q1 = select(subq2).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        self.assert_compile(q1.order_by(subq2.c.anon_2_t1_c1), 'SELECT anon_1.anon_2_t1_c1 AS anon_1_anon_2_t1_c1, anon_1.anon_2_t1_c2 AS anon_1_anon_2_t1_c2 FROM (SELECT anon_2.t1_c1 AS anon_2_t1_c1, anon_2.t1_c2 AS anon_2_t1_c2 FROM (SELECT t1.c1 AS t1_c1, t1.c2 AS t1_c2 FROM t1 WHERE t1.c1 = :c1_1) AS anon_2) AS anon_1 ORDER BY anon_1.anon_2_t1_c1')

    def test_anonymous_expression_from_self_twice_newstyle_wlabels(self):
        if False:
            for i in range(10):
                print('nop')
        from sqlalchemy.sql import column
        (c1, c2) = (column('c1'), column('c2'))
        subq = select(c1, c2).where(c1 == 'dog').subquery()
        subq2 = select(subq).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        stmt = select(subq2).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        self.assert_compile(stmt.order_by(subq2.c.corresponding_column(c1)), 'SELECT anon_1.anon_2_c1 AS anon_1_anon_2_c1, anon_1.anon_2_c2 AS anon_1_anon_2_c2 FROM (SELECT anon_2.c1 AS anon_2_c1, anon_2.c2 AS anon_2_c2 FROM (SELECT c1, c2 WHERE c1 = :c1_1) AS anon_2) AS anon_1 ORDER BY anon_1.anon_2_c1')

    def test_anonymous_expression_from_self_twice_newstyle_wolabels(self):
        if False:
            print('Hello World!')
        from sqlalchemy.sql import column
        (c1, c2) = (column('c1'), column('c2'))
        subq = select(c1, c2).where(c1 == 'dog').subquery()
        subq2 = select(subq).subquery()
        stmt = select(subq2)
        self.assert_compile(stmt.order_by(subq2.c.c1), 'SELECT anon_1.c1, anon_1.c2 FROM (SELECT anon_2.c1 AS c1, anon_2.c2 AS c2 FROM (SELECT c1, c2 WHERE c1 = :c1_1) AS anon_2) AS anon_1 ORDER BY anon_1.c1')

    def test_anonymous_labeled_expression_oldstyle(self):
        if False:
            for i in range(10):
                print('nop')
        sess = fixture_session()
        (c1, c2) = (column('c1'), column('c2'))
        q1 = sess.query(c1.label('foo'), c2.label('bar')).filter(c1 == 'dog')
        q2 = sess.query(c1.label('foo'), c2.label('bar')).filter(c1 == 'cat')
        q3 = q1.union(q2)
        self.assert_compile(q3.order_by(c1), 'SELECT anon_1.foo AS anon_1_foo, anon_1.bar AS anon_1_bar FROM (SELECT c1 AS foo, c2 AS bar WHERE c1 = :c1_1 UNION SELECT c1 AS foo, c2 AS bar WHERE c1 = :c1_2) AS anon_1 ORDER BY anon_1.foo')

    def test_anonymous_labeled_expression_newstyle(self):
        if False:
            while True:
                i = 10
        (c1, c2) = (column('c1'), column('c2'))
        q1 = select(c1.label('foo'), c2.label('bar')).where(c1 == 'dog')
        q2 = select(c1.label('foo'), c2.label('bar')).where(c1 == 'cat')
        subq = union(q1, q2).subquery()
        q3 = select(subq).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        self.assert_compile(q3.order_by(subq.c.foo), 'SELECT anon_1.foo AS anon_1_foo, anon_1.bar AS anon_1_bar FROM (SELECT c1 AS foo, c2 AS bar WHERE c1 = :c1_1 UNION SELECT c1 AS foo, c2 AS bar WHERE c1 = :c1_2) AS anon_1 ORDER BY anon_1.foo')

    def test_anonymous_expression_plus_flag_aliased_join_newstyle(self):
        if False:
            for i in range(10):
                print('nop')
        User = self.classes.User
        Address = self.classes.Address
        addresses = self.tables.addresses
        sess = fixture_session()
        q1 = sess.query(User.id).filter(User.id > 5)
        uq = aliased(User, q1.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery())
        aa = aliased(Address)
        q1 = sess.query(uq.id).join(uq.addresses.of_type(aa)).order_by(uq.id, aa.id, addresses.c.id)
        self.assert_compile(q1, 'SELECT anon_1.users_id AS anon_1_users_id FROM (SELECT users.id AS users_id FROM users WHERE users.id > :id_1) AS anon_1 JOIN addresses AS addresses_1 ON anon_1.users_id = addresses_1.user_id ORDER BY anon_1.users_id, addresses_1.id, addresses.id')

    def test_anonymous_expression_plus_explicit_aliased_join_newstyle(self):
        if False:
            return 10
        "test that the 'dont alias non-ORM' rule remains for other\n        kinds of aliasing when _from_selectable() is used."
        User = self.classes.User
        Address = self.classes.Address
        addresses = self.tables.addresses
        sess = fixture_session()
        q1 = sess.query(User.id).filter(User.id > 5).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        uq = aliased(User, q1)
        aa = aliased(Address)
        q1 = sess.query(uq.id).join(aa, uq.addresses).order_by(uq.id, aa.id, addresses.c.id)
        self.assert_compile(q1, 'SELECT anon_1.users_id AS anon_1_users_id FROM (SELECT users.id AS users_id FROM users WHERE users.id > :id_1) AS anon_1 JOIN addresses AS addresses_1 ON anon_1.users_id = addresses_1.user_id ORDER BY anon_1.users_id, addresses_1.id, addresses.id')

class AddEntityEquivalenceTest(fixtures.MappedTest, AssertsCompiledSQL):
    run_setup_mappers = 'once'

    @classmethod
    def define_tables(cls, metadata):
        if False:
            for i in range(10):
                print('nop')
        Table('a', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('name', String(50)), Column('type', String(20)), Column('bid', Integer, ForeignKey('b.id')))
        Table('b', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('name', String(50)), Column('type', String(20)))
        Table('c', metadata, Column('id', Integer, ForeignKey('b.id'), primary_key=True), Column('age', Integer))
        Table('d', metadata, Column('id', Integer, ForeignKey('a.id'), primary_key=True), Column('dede', Integer))

    @classmethod
    def setup_classes(cls):
        if False:
            for i in range(10):
                print('nop')

        class A(cls.Comparable):
            pass

        class B(cls.Comparable):
            pass

        class C(B):
            pass

        class D(A):
            pass

    @classmethod
    def setup_mappers(cls):
        if False:
            i = 10
            return i + 15
        (a, c, b, d) = (cls.tables.a, cls.tables.c, cls.tables.b, cls.tables.d)
        (A, B, C, D) = cls.classes('A', 'B', 'C', 'D')
        cls.mapper_registry.map_imperatively(A, a, polymorphic_identity='a', polymorphic_on=a.c.type, with_polymorphic=('*', None), properties={'link': relationship(B, uselist=False, backref='back')})
        cls.mapper_registry.map_imperatively(B, b, polymorphic_identity='b', polymorphic_on=b.c.type, with_polymorphic=('*', None))
        cls.mapper_registry.map_imperatively(C, c, inherits=B, polymorphic_identity='c')
        cls.mapper_registry.map_imperatively(D, d, inherits=A, polymorphic_identity='d')

    @classmethod
    def insert_data(cls, connection):
        if False:
            print('Hello World!')
        (A, C, B) = (cls.classes.A, cls.classes.C, cls.classes.B)
        sess = Session(connection)
        sess.add_all([B(name='b1'), A(name='a1', link=C(name='c1', age=3)), C(name='c2', age=6), A(name='a2')])
        sess.flush()

    def test_add_entity_equivalence(self):
        if False:
            return 10
        (A, C, B) = (self.classes.A, self.classes.C, self.classes.B)
        sess = fixture_session()
        for q in [sess.query(A, B).join(A.link), sess.query(A).join(A.link).add_entity(B)]:
            eq_(q.all(), [(A(bid=2, id=1, name='a1', type='a'), C(age=3, id=2, name='c1', type='c'))])
        for q in [sess.query(B, A).join(B.back), sess.query(B).join(B.back).add_entity(A), sess.query(B).add_entity(A).join(B.back)]:
            eq_(q.all(), [(C(age=3, id=2, name='c1', type='c'), A(bid=2, id=1, name='a1', type='a'))])

class InstancesTest(QueryTest, AssertsCompiledSQL):

    def test_from_alias_two_needs_nothing(self):
        if False:
            while True:
                i = 10
        (User, addresses, users) = (self.classes.User, self.tables.addresses, self.tables.users)
        query = users.select().where(users.c.id == 7).union(users.select().where(users.c.id > 7)).alias('ulist').outerjoin(addresses).select().order_by(text('ulist.id'), addresses.c.id)
        sess = fixture_session()
        q = sess.query(User)

        def go():
            if False:
                return 10
            result = q.options(contains_eager(User.addresses)).from_statement(query).all()
            assert self.static.user_address_result == result
        self.assert_sql_count(testing.db, go, 1)

    def test_from_alias_two(self):
        if False:
            i = 10
            return i + 15
        (User, addresses, users) = (self.classes.User, self.tables.addresses, self.tables.users)
        query = users.select().where(users.c.id == 7).union(users.select().where(users.c.id > 7)).alias('ulist').outerjoin(addresses).select().order_by(text('ulist.id'), addresses.c.id)
        sess = fixture_session()

        def go():
            if False:
                for i in range(10):
                    print('nop')
            ulist = query.alias('ulist')
            ulist_alias = aliased(User, alias=ulist)
            q = sess.query(ulist_alias)
            result = q.options(contains_eager(ulist_alias.addresses, alias=ulist)).all()
            assert self.static.user_address_result == result
        self.assert_sql_count(testing.db, go, 1)

    def test_from_alias_three(self):
        if False:
            print('Hello World!')
        (User, addresses, users) = (self.classes.User, self.tables.addresses, self.tables.users)
        query = users.select().where(users.c.id == 7).union(users.select().where(users.c.id > 7)).alias('ulist').outerjoin(addresses).select().order_by(text('ulist.id'), addresses.c.id)
        sess = fixture_session()

        def go():
            if False:
                return 10
            qs = query.subquery()
            ua = aliased(User, qs)
            result = sess.query(ua).options(contains_eager(ua.addresses, alias=qs)).all()
            assert self.static.user_address_result == result
        self.assert_sql_count(testing.db, go, 1)

    def test_from_alias_four(self):
        if False:
            print('Hello World!')
        (User, addresses, users) = (self.classes.User, self.tables.addresses, self.tables.users)
        Address = self.classes.Address
        sess = fixture_session()
        adalias = addresses.alias()
        query = users.select().where(users.c.id == 7).union(users.select().where(users.c.id > 7)).alias('ulist').outerjoin(adalias).select().order_by(text('ulist.id'), adalias.c.id)

        def go():
            if False:
                return 10
            qs = query.subquery()
            ua = aliased(User, qs)
            aa = aliased(Address, qs)
            result = sess.query(ua).options(contains_eager(ua.addresses.of_type(aa))).all()
            assert self.static.user_address_result == result
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_one(self):
        if False:
            print('Hello World!')
        (addresses, User) = (self.tables.addresses, self.classes.User)
        sess = fixture_session()
        q = sess.query(User).outerjoin(User.addresses).options(contains_eager(User.addresses)).order_by(User.id, addresses.c.id)
        self.assert_compile(q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).statement, 'SELECT addresses.id AS addresses_id, addresses.user_id AS addresses_user_id, addresses.email_address AS addresses_email_address, users.id AS users_id, users.name AS users_name FROM users LEFT OUTER JOIN addresses ON users.id = addresses.user_id ORDER BY users.id, addresses.id', dialect=default.DefaultDialect())

        def go():
            if False:
                while True:
                    i = 10
            assert self.static.user_address_result == q.all()
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_two(self):
        if False:
            for i in range(10):
                print('nop')
        (users, addresses, User) = (self.tables.users, self.tables.addresses, self.classes.User)
        sess = fixture_session()
        adalias = addresses.alias()
        qq = users.outerjoin(adalias)
        ua = aliased(User, qq)
        q = sess.query(ua).options(contains_eager(ua.addresses, alias=adalias)).order_by(User.id, adalias.c.id)

        def go():
            if False:
                i = 10
                return i + 15
            eq_(self.static.user_address_result, q.all())
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_four(self):
        if False:
            print('Hello World!')
        (users, addresses, User) = (self.tables.users, self.tables.addresses, self.classes.User)
        sess = fixture_session()
        selectquery = users.outerjoin(addresses).select().where(users.c.id < 10).order_by(users.c.id, addresses.c.id)
        q = sess.query(User)

        def go():
            if False:
                while True:
                    i = 10
            result = q.options(contains_eager(User.addresses)).from_statement(selectquery).all()
            assert self.static.user_address_result[0:3] == result
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_four_future(self):
        if False:
            print('Hello World!')
        (users, addresses, User) = (self.tables.users, self.tables.addresses, self.classes.User)
        sess = fixture_session(future=True)
        selectquery = users.outerjoin(addresses).select().where(users.c.id < 10).order_by(users.c.id, addresses.c.id)
        q = select(User)

        def go():
            if False:
                i = 10
                return i + 15
            result = sess.execute(q.options(contains_eager(User.addresses)).from_statement(selectquery)).scalars().unique().all()
            assert self.static.user_address_result[0:3] == result
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_aliased(self):
        if False:
            for i in range(10):
                print('nop')
        (User, Address) = (self.classes.User, self.classes.Address)
        sess = fixture_session()
        q = sess.query(User)
        adalias = aliased(Address)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            result = q.options(contains_eager(User.addresses.of_type(adalias))).outerjoin(adalias, User.addresses).order_by(User.id, adalias.id)
            assert self.static.user_address_result == result.all()
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_multi_alias(self):
        if False:
            return 10
        (orders, items, users, order_items, User) = (self.tables.orders, self.tables.items, self.tables.users, self.tables.order_items, self.classes.User)
        Order = self.classes.Order
        sess = fixture_session()
        q = sess.query(User)
        oalias = orders.alias('o1')
        ialias = items.alias('i1')
        query = users.outerjoin(oalias).outerjoin(order_items).outerjoin(ialias).select().order_by(users.c.id, oalias.c.id, ialias.c.id)

        def go():
            if False:
                print('Hello World!')
            result = list(q.options(contains_eager(User.orders, alias=oalias).contains_eager(Order.items, alias=ialias)).from_statement(query))
            assert self.static.user_order_result == result
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_multi_aliased(self):
        if False:
            print('Hello World!')
        (Item, User, Order) = (self.classes.Item, self.classes.User, self.classes.Order)
        sess = fixture_session()
        q = sess.query(User)
        oalias = aliased(Order)
        ialias = aliased(Item)

        def go():
            if False:
                return 10
            result = q.options(contains_eager(User.orders, alias=oalias), contains_eager(User.orders, Order.items, alias=ialias)).outerjoin(oalias, User.orders).outerjoin(ialias, oalias.items).order_by(User.id, oalias.id, ialias.id)
            assert self.static.user_order_result == result.all()
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_multi_aliased_of_type(self):
        if False:
            while True:
                i = 10
        (Item, User, Order) = (self.classes.Item, self.classes.User, self.classes.Order)
        sess = fixture_session()
        q = sess.query(User)
        oalias = aliased(Order)
        ialias = aliased(Item)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            result = q.options(contains_eager(User.orders.of_type(oalias)).contains_eager(oalias.items.of_type(ialias))).outerjoin(User.orders.of_type(oalias)).outerjoin(oalias.items.of_type(ialias)).order_by(User.id, oalias.id, ialias.id)
            assert self.static.user_order_result == result.all()
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_chaining(self):
        if False:
            i = 10
            return i + 15
        "test that contains_eager() 'chains' by default."
        (Dingaling, User, Address) = (self.classes.Dingaling, self.classes.User, self.classes.Address)
        sess = fixture_session()
        q = sess.query(User).join(User.addresses).join(Address.dingaling).options(contains_eager(User.addresses, Address.dingaling))

        def go():
            if False:
                while True:
                    i = 10
            eq_(q.all(), [User(name='ed', addresses=[Address(email_address='ed@wood.com', dingaling=Dingaling(data='ding 1/2'))]), User(name='fred', addresses=[Address(email_address='fred@fred.com', dingaling=Dingaling(data='ding 2/5'))])])
        self.assert_sql_count(testing.db, go, 1)

    def test_contains_eager_chaining_aliased_endpoint(self):
        if False:
            while True:
                i = 10
        "test that contains_eager() 'chains' by default and supports\n        an alias at the end."
        (Dingaling, User, Address) = (self.classes.Dingaling, self.classes.User, self.classes.Address)
        sess = fixture_session()
        da = aliased(Dingaling, name='foob')
        q = sess.query(User).join(User.addresses).join(da, Address.dingaling).options(contains_eager(User.addresses, Address.dingaling, alias=da))

        def go():
            if False:
                for i in range(10):
                    print('nop')
            eq_(q.all(), [User(name='ed', addresses=[Address(email_address='ed@wood.com', dingaling=Dingaling(data='ding 1/2'))]), User(name='fred', addresses=[Address(email_address='fred@fred.com', dingaling=Dingaling(data='ding 2/5'))])])
        self.assert_sql_count(testing.db, go, 1)

    def test_mixed_eager_contains_with_limit(self):
        if False:
            i = 10
            return i + 15
        (Order, User, Address) = (self.classes.Order, self.classes.User, self.classes.Address)
        sess = fixture_session()
        q = sess.query(User)

        def go():
            if False:
                print('Hello World!')
            result = q.outerjoin(User.orders).options(joinedload(User.addresses), contains_eager(User.orders)).order_by(User.id, Order.id).offset(1).limit(2).all()
            eq_(result, [User(id=7, addresses=[Address(email_address='jack@bean.com', user_id=7, id=1)], name='jack', orders=[Order(address_id=1, user_id=7, description='order 3', isopen=1, id=3), Order(address_id=None, user_id=7, description='order 5', isopen=0, id=5)])])
        self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()

        def go():
            if False:
                return 10
            oalias = aliased(Order)
            result = q.outerjoin(oalias, User.orders).options(joinedload(User.addresses), contains_eager(User.orders, alias=oalias)).order_by(User.id, oalias.id).offset(1).limit(2).all()
            eq_(result, [User(id=7, addresses=[Address(email_address='jack@bean.com', user_id=7, id=1)], name='jack', orders=[Order(address_id=1, user_id=7, description='order 3', isopen=1, id=3), Order(address_id=None, user_id=7, description='order 5', isopen=0, id=5)])])
        self.assert_sql_count(testing.db, go, 1)

class MixedEntitiesTest(QueryTest, AssertsCompiledSQL):
    __dialect__ = 'default'

    def test_alias_naming(self):
        if False:
            i = 10
            return i + 15
        User = self.classes.User
        sess = fixture_session()
        ua = aliased(User, name='foobar')
        q = sess.query(ua)
        self.assert_compile(q, 'SELECT foobar.id AS foobar_id, foobar.name AS foobar_name FROM users AS foobar')

    def test_correlated_subquery(self):
        if False:
            return 10
        "test that a subquery constructed from ORM attributes doesn't leak\n        out those entities to the outermost query."
        (Address, users, User) = (self.classes.Address, self.tables.users, self.classes.User)
        sess = fixture_session()
        subq = select(func.count()).where(User.id == Address.user_id).correlate(users).label('count')
        eq_(list(sess.query(User, subq)[0:3]), [(User(id=7, name='jack'), 1), (User(id=8, name='ed'), 3), (User(id=9, name='fred'), 1)])
        subq = select(func.count()).where(User.id == Address.user_id).label('count')
        eq_(list(sess.query(User, subq)[0:3]), [(User(id=7, name='jack'), 1), (User(id=8, name='ed'), 3), (User(id=9, name='fred'), 1)])

    @testing.combinations((True,), (False,))
    def test_no_uniquing_cols_legacy(self, with_entities):
        if False:
            return 10
        'test #6924'
        User = self.classes.User
        Address = self.classes.Address
        sess = fixture_session()
        if with_entities:
            q = sess.query(User).join(Address).filter(Address.user_id == 8).with_entities(User.id, User.name).order_by(User.id)
        else:
            q = sess.query(User.id, User.name).join(Address).filter(Address.user_id == 8).order_by(User.id)
        is_(q._compile_state()._primary_entity, None)
        eq_(q.all(), [(8, 'ed'), (8, 'ed'), (8, 'ed')])

    @testing.combinations((True,), (False,))
    def test_no_uniquing_cols(self, with_entities):
        if False:
            while True:
                i = 10
        'test #6924'
        User = self.classes.User
        Address = self.classes.Address
        if with_entities:
            stmt = select(User).join(Address).filter(Address.user_id == 8).with_only_columns(User.id, User.name).order_by(User.id)
        else:
            stmt = select(User.id, User.name).join(Address).filter(Address.user_id == 8).order_by(User.id)
        compile_state = ORMSelectCompileState.create_for_statement(stmt, None)
        is_(compile_state._primary_entity, None)

    def test_column_queries_one(self):
        if False:
            print('Hello World!')
        User = self.classes.User
        sess = fixture_session()
        eq_(sess.query(User.name).all(), [('jack',), ('ed',), ('fred',), ('chuck',)])

    def test_column_queries_two(self):
        if False:
            return 10
        (users, User) = (self.tables.users, self.classes.User)
        sess = fixture_session()
        sel = users.select().where(User.id.in_([7, 8])).alias()
        ua = aliased(User, sel)
        q = sess.query(ua.name)
        q2 = q.all()
        eq_(list(q2), [('jack',), ('ed',)])

    def test_column_queries_three(self):
        if False:
            i = 10
            return i + 15
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        eq_(sess.query(User.name, Address.email_address).filter(User.id == Address.user_id).all(), [('jack', 'jack@bean.com'), ('ed', 'ed@wood.com'), ('ed', 'ed@bettyboop.com'), ('ed', 'ed@lala.com'), ('fred', 'fred@fred.com')])

    def test_column_queries_four(self):
        if False:
            for i in range(10):
                print('nop')
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        eq_(sess.query(User.name, func.count(Address.email_address)).outerjoin(User.addresses).group_by(User.id, User.name).order_by(User.id).all(), [('jack', 1), ('ed', 3), ('fred', 1), ('chuck', 0)])

    def test_column_queries_five(self):
        if False:
            print('Hello World!')
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        eq_(sess.query(User, func.count(Address.email_address)).outerjoin(User.addresses).group_by(User).order_by(User.id).all(), [(User(name='jack', id=7), 1), (User(name='ed', id=8), 3), (User(name='fred', id=9), 1), (User(name='chuck', id=10), 0)])

    def test_column_queries_six(self):
        if False:
            for i in range(10):
                print('nop')
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        eq_(sess.query(func.count(Address.email_address), User).outerjoin(User.addresses).group_by(User).order_by(User.id).all(), [(1, User(name='jack', id=7)), (3, User(name='ed', id=8)), (1, User(name='fred', id=9)), (0, User(name='chuck', id=10))])

    def test_column_queries_seven(self):
        if False:
            while True:
                i = 10
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        adalias = aliased(Address)
        eq_(sess.query(User, func.count(adalias.email_address)).outerjoin(User.addresses.of_type(adalias)).group_by(User).order_by(User.id).all(), [(User(name='jack', id=7), 1), (User(name='ed', id=8), 3), (User(name='fred', id=9), 1), (User(name='chuck', id=10), 0)])

    def test_column_queries_eight(self):
        if False:
            i = 10
            return i + 15
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        adalias = aliased(Address)
        eq_(sess.query(func.count(adalias.email_address), User).outerjoin(adalias, User.addresses).group_by(User).order_by(User.id).all(), [(1, User(name='jack', id=7)), (3, User(name='ed', id=8)), (1, User(name='fred', id=9)), (0, User(name='chuck', id=10))])

    def test_column_queries_nine(self):
        if False:
            return 10
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        adalias = aliased(Address)
        subq = sess.query(User, adalias.email_address, adalias.id).outerjoin(adalias, User.addresses).subquery()
        ua = aliased(User, subq)
        aa = aliased(adalias, subq)
        q = sess.query(ua, aa.email_address).order_by(ua.id, aa.id)
        eq_(q.all(), [(User(name='jack', id=7), 'jack@bean.com'), (User(name='ed', id=8), 'ed@wood.com'), (User(name='ed', id=8), 'ed@bettyboop.com'), (User(name='ed', id=8), 'ed@lala.com'), (User(name='fred', id=9), 'fred@fred.com'), (User(name='chuck', id=10), None)])

    def test_column_queries_ten(self):
        if False:
            while True:
                i = 10
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        aa = aliased(Address)
        subq = sess.query(User).join(aa, User.addresses).filter(aa.email_address.like('%ed%')).subquery()
        ua = aliased(User, subq)
        eq_(sess.query(ua).all(), [User(name='ed', id=8), User(name='fred', id=9)])

    def test_column_queries_eleven(self):
        if False:
            while True:
                i = 10
        (Address, User) = (self.classes.Address, self.classes.User)
        sess = fixture_session()
        adalias = aliased(Address)
        q1 = sess.query(User, adalias.email_address).outerjoin(adalias, User.addresses).options(joinedload(User.addresses)).order_by(User.id, adalias.id).limit(10)
        subq = sess.query(User, adalias.email_address, adalias.id).outerjoin(adalias, User.addresses).subquery()
        ua = aliased(User, subq)
        aa = aliased(adalias, subq)
        q2 = sess.query(ua, aa.email_address).options(joinedload(ua.addresses)).order_by(ua.id, aa.id).limit(10)
        for q in [q1, q2]:
            eq_(q.all(), [(User(addresses=[Address(user_id=7, email_address='jack@bean.com', id=1)], name='jack', id=7), 'jack@bean.com'), (User(addresses=[Address(user_id=8, email_address='ed@wood.com', id=2), Address(user_id=8, email_address='ed@bettyboop.com', id=3), Address(user_id=8, email_address='ed@lala.com', id=4)], name='ed', id=8), 'ed@wood.com'), (User(addresses=[Address(user_id=8, email_address='ed@wood.com', id=2), Address(user_id=8, email_address='ed@bettyboop.com', id=3), Address(user_id=8, email_address='ed@lala.com', id=4)], name='ed', id=8), 'ed@bettyboop.com'), (User(addresses=[Address(user_id=8, email_address='ed@wood.com', id=2), Address(user_id=8, email_address='ed@bettyboop.com', id=3), Address(user_id=8, email_address='ed@lala.com', id=4)], name='ed', id=8), 'ed@lala.com'), (User(addresses=[Address(user_id=9, email_address='fred@fred.com', id=5)], name='fred', id=9), 'fred@fred.com'), (User(addresses=[], name='chuck', id=10), None)])

    def test_column_from_limited_joinedload(self):
        if False:
            return 10
        User = self.classes.User
        sess = fixture_session()

        def go():
            if False:
                i = 10
                return i + 15
            results = sess.query(User).limit(1).options(joinedload(User.addresses)).add_columns(User.name).all()
            eq_(results, [(User(name='jack'), 'jack')])
        self.assert_sql_count(testing.db, go, 1)

    def test_self_referential_from_self(self):
        if False:
            return 10
        Order = self.classes.Order
        sess = fixture_session()
        oalias = aliased(Order)
        q1 = sess.query(Order, oalias).filter(Order.user_id == oalias.user_id).filter(Order.user_id == 7).filter(Order.id > oalias.id).order_by(Order.id, oalias.id)
        subq = sess.query(Order, oalias).filter(Order.id > oalias.id).subquery()
        (oa, oaa) = (aliased(Order, subq), aliased(oalias, subq))
        q2 = sess.query(oa, oaa).filter(oa.user_id == oaa.user_id).filter(oa.user_id == 7).order_by(oa.id, oaa.id)
        subq = sess.query(oalias, Order).filter(Order.id < oalias.id).subquery()
        (oa, oaa) = (aliased(Order, subq), aliased(oalias, subq))
        q3 = sess.query(oaa, oa).filter(oaa.user_id == oa.user_id).filter(oaa.user_id == 7).order_by(oaa.id, oa.id)
        subq = sess.query(Order, oalias).filter(Order.user_id == oalias.user_id).filter(Order.user_id == 7).filter(Order.id > oalias.id).subquery()
        (oa, oaa) = (aliased(Order, subq), aliased(oalias, subq))
        q4 = sess.query(oa, oaa).order_by(oa.id, oaa.id).limit(10).options(joinedload(oa.items))
        subq4 = subq
        for i in range(4):
            (oa, oaa) = (aliased(Order, subq4), aliased(oaa, subq4))
            subq4 = sess.query(oa, oaa).subquery()
        (oa, oaa) = (aliased(Order, subq4), aliased(oaa, subq4))
        q5 = sess.query(oa, oaa).order_by(oa.id, oaa.id).limit(10).options(joinedload(oa.items))
        for q in [q1, q2, q3, q4, q5]:
            eq_(q.all(), [(Order(address_id=1, description='order 3', isopen=1, user_id=7, id=3), Order(address_id=1, description='order 1', isopen=0, user_id=7, id=1)), (Order(address_id=None, description='order 5', isopen=0, user_id=7, id=5), Order(address_id=1, description='order 1', isopen=0, user_id=7, id=1)), (Order(address_id=None, description='order 5', isopen=0, user_id=7, id=5), Order(address_id=1, description='order 3', isopen=1, user_id=7, id=3))])

    def test_from_self_internal_literals_newstyle(self):
        if False:
            for i in range(10):
                print('nop')
        Order = self.classes.Order
        stmt = select(Order.id, Order.description, literal_column("'q'").label('foo')).where(Order.description == 'order 3')
        subq = aliased(Order, stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery())
        stmt = select(subq).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        self.assert_compile(stmt, "SELECT anon_1.orders_id AS anon_1_orders_id, anon_1.orders_description AS anon_1_orders_description FROM (SELECT orders.id AS orders_id, orders.description AS orders_description, 'q' AS foo FROM orders WHERE orders.description = :description_1) AS anon_1")

    def test_multi_mappers(self):
        if False:
            i = 10
            return i + 15
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        test_session = fixture_session()
        (user7, user8, user9, user10) = test_session.query(User).all()
        (address1, address2, address3, address4, address5) = test_session.query(Address).all()
        expected = [(user7, address1), (user8, address2), (user8, address3), (user8, address4), (user9, address5), (user10, None)]
        sess = fixture_session(future=True)
        selectquery = users.outerjoin(addresses).select().order_by(users.c.id, addresses.c.id)
        result = sess.execute(select(User, Address).from_statement(selectquery))
        eq_(list(result), expected)
        sess.expunge_all()
        for address_entity in (Address, aliased(Address)):
            q = sess.query(User).add_entity(address_entity).outerjoin(address_entity, User.addresses).order_by(User.id, address_entity.id)
            eq_(q.all(), expected)
            sess.expunge_all()
            q = sess.query(User).add_entity(address_entity)
            q = q.join(address_entity, User.addresses)
            q = q.filter_by(email_address='ed@bettyboop.com')
            eq_(q.all(), [(user8, address3)])
            sess.expunge_all()
            q = sess.query(User, address_entity).join(address_entity, User.addresses).filter_by(email_address='ed@bettyboop.com')
            eq_(q.all(), [(user8, address3)])
            sess.expunge_all()
            q = sess.query(User, address_entity).join(address_entity, User.addresses).options(joinedload(User.addresses)).filter_by(email_address='ed@bettyboop.com')
            eq_(list(util.OrderedSet(q.all())), [(user8, address3)])
            sess.expunge_all()

    def test_aliased_multi_mappers(self):
        if False:
            return 10
        (User, addresses, users, Address) = (self.classes.User, self.tables.addresses, self.tables.users, self.classes.Address)
        sess = fixture_session()
        (user7, user8, user9, user10) = sess.query(User).all()
        (address1, address2, address3, address4, address5) = sess.query(Address).all()
        expected = [(user7, address1), (user8, address2), (user8, address3), (user8, address4), (user9, address5), (user10, None)]
        adalias = addresses.alias('adalias')
        uaj = users.outerjoin(adalias)
        ua = aliased(User, uaj)
        q = sess.query(ua)
        q = q.add_entity(Address, alias=adalias)
        result = q.order_by(User.id, adalias.c.id).all()
        assert result == expected
        sess.expunge_all()
        q = sess.query(ua).add_entity(Address, alias=adalias)
        result = q.filter(adalias.c.email_address == 'ed@bettyboop.com').all()
        assert result == [(user8, address3)]

    def test_with_entities(self):
        if False:
            while True:
                i = 10
        (User, Address) = (self.classes.User, self.classes.Address)
        sess = fixture_session()
        q = sess.query(User).filter(User.id == 7).order_by(User.name)
        self.assert_compile(q.with_entities(User.id, Address).filter(Address.user_id == User.id), 'SELECT users.id AS users_id, addresses.id AS addresses_id, addresses.user_id AS addresses_user_id, addresses.email_address AS addresses_email_address FROM users, addresses WHERE users.id = :id_1 AND addresses.user_id = users.id ORDER BY users.name')

    def test_multi_columns(self):
        if False:
            i = 10
            return i + 15
        (users, User) = (self.tables.users, self.classes.User)
        sess = fixture_session()
        expected = [(u, u.name) for u in sess.query(User).all()]
        for add_col in (User.name, users.c.name):
            assert sess.query(User).add_columns(add_col).all() == expected
            sess.expunge_all()
        assert_raises(sa_exc.ArgumentError, sess.query(User).add_columns, object())

    def test_add_multi_columns(self):
        if False:
            print('Hello World!')
        'test that add_column accepts a FROM clause.'
        (users, User) = (self.tables.users, self.classes.User)
        sess = fixture_session()
        eq_(sess.query(User.id).add_columns(users).all(), [(7, 7, 'jack'), (8, 8, 'ed'), (9, 9, 'fred'), (10, 10, 'chuck')])

    def test_multi_columns_2(self):
        if False:
            for i in range(10):
                print('nop')
        'test aliased/nonalised joins with the usage of add_columns()'
        (User, Address, addresses, users) = (self.classes.User, self.classes.Address, self.tables.addresses, self.tables.users)
        sess = fixture_session()
        (user7, user8, user9, user10) = sess.query(User).all()
        expected = [(user7, 1), (user8, 3), (user9, 1), (user10, 0)]
        q = sess.query(User)
        q = q.group_by(users).order_by(User.id).outerjoin(User.addresses).add_columns(func.count(Address.id).label('count'))
        eq_(q.all(), expected)
        sess.expunge_all()
        adalias = aliased(Address)
        q = sess.query(User)
        q = q.group_by(users).order_by(User.id).outerjoin(User.addresses.of_type(adalias)).add_columns(func.count(adalias.id).label('count'))
        eq_(q.all(), expected)
        sess.expunge_all()
        count = func.count(addresses.c.id).label('count')
        s = select(users, count).select_from(users.outerjoin(addresses)).group_by(*[c for c in users.c]).order_by(User.id)
        q = sess.query(User)
        result = q.add_columns(s.selected_columns.count).from_statement(s).all()
        assert result == expected

    def test_multi_columns_3(self):
        if False:
            while True:
                i = 10
        User = self.classes.User
        users = self.tables.users
        sess = fixture_session()
        q = sess.query(User.id, User.name)
        stmt = select(users).order_by(users.c.id)
        q = q.from_statement(stmt)
        eq_(q.all(), [(7, 'jack'), (8, 'ed'), (9, 'fred'), (10, 'chuck')])

    def test_raw_columns(self):
        if False:
            return 10
        (addresses, users, User) = (self.tables.addresses, self.tables.users, self.classes.User)
        sess = fixture_session()
        (user7, user8, user9, user10) = sess.query(User).all()
        expected = [(user7, 1, 'Name:jack'), (user8, 3, 'Name:ed'), (user9, 1, 'Name:fred'), (user10, 0, 'Name:chuck')]
        adalias = addresses.alias()
        with fixture_session() as sess:
            q = sess.query(User).add_columns(func.count(adalias.c.id), 'Name:' + users.c.name).outerjoin(adalias).group_by(users).order_by(users.c.id)
            eq_(q.all(), expected)
        s = select(users, func.count(addresses.c.id).label('count'), ('Name:' + users.c.name).label('concat')).select_from(users.outerjoin(addresses)).group_by(*[c for c in users.c]).order_by(users.c.id)
        with fixture_session() as sess:
            q = sess.query(User)
            result = q.add_columns(s.selected_columns.count, s.selected_columns.concat).from_statement(s).all()
            eq_(result, expected)
        with fixture_session() as sess:
            uaj = users.outerjoin(addresses)
            ua = aliased(User, uaj)
            q = fixture_session().query(ua).add_columns(func.count(addresses.c.id), 'Name:' + users.c.name).group_by(users).order_by(users.c.id)
            eq_(q.all(), expected)
        with fixture_session() as sess:
            q = sess.query(User).add_columns(func.count(addresses.c.id), 'Name:' + users.c.name).outerjoin(User.addresses).group_by(users).order_by(users.c.id)
            eq_(q.all(), expected)
        with fixture_session() as sess:
            q = sess.query(User).add_columns(func.count(adalias.c.id), 'Name:' + users.c.name).outerjoin(adalias).group_by(users).order_by(users.c.id)
            eq_(q.all(), expected)

    def test_unrelated_column(self):
        if False:
            print('Hello World!')
        'Test for #9217'
        User = self.classes.User
        q = select(User.id, func.lower('SANDY').label('name')).where(User.id == 7)
        s = select(User).from_statement(q)
        sess = fixture_session()
        res = sess.scalars(s).one()
        in_('name', res.__dict__)
        eq_(res, User(name='sandy', id=7))

    def test_unrelated_column_col_prop(self, decl_base):
        if False:
            return 10
        'Test for #9217 combined with #9273'

        class User(ComparableEntity, decl_base):
            __tablename__ = 'some_user_table'
            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str] = mapped_column()
            age: Mapped[int] = mapped_column()
            is_adult: Mapped[bool] = column_property(age >= 18)
        stmt = select(literal(1).label('id'), literal('John').label('name'), literal(30).label('age'))
        s = select(User).from_statement(stmt)
        sess = fixture_session()
        res = sess.scalars(s).one()
        eq_(res, User(name='John', age=30, id=1))

    def test_expression_selectable_matches_mzero(self):
        if False:
            i = 10
            return i + 15
        (User, Address) = (self.classes.User, self.classes.Address)
        ua = aliased(User)
        aa = aliased(Address)
        s = fixture_session()
        for (crit, j, exp) in [(User.id + Address.id, (User.addresses,), 'SELECT users.id + addresses.id AS anon_1 FROM users JOIN addresses ON users.id = addresses.user_id'), (User.id + Address.id, (Address.user,), 'SELECT users.id + addresses.id AS anon_1 FROM addresses JOIN users ON users.id = addresses.user_id'), (Address.id + User.id, (User.addresses,), 'SELECT addresses.id + users.id AS anon_1 FROM users JOIN addresses ON users.id = addresses.user_id'), (User.id + aa.id, (aa, User.addresses), 'SELECT users.id + addresses_1.id AS anon_1 FROM users JOIN addresses AS addresses_1 ON users.id = addresses_1.user_id')]:
            q = s.query(crit)
            mzero = q._compile_state()._entity_zero()
            is_(mzero, q._compile_state()._entities[0].entity_zero)
            q = q.join(*j)
            self.assert_compile(q, exp)
        for (crit, j, exp) in [(ua.id + Address.id, (ua.addresses,), 'SELECT users_1.id + addresses.id AS anon_1 FROM users AS users_1 JOIN addresses ON users_1.id = addresses.user_id'), (ua.id + aa.id, (aa, ua.addresses), 'SELECT users_1.id + addresses_1.id AS anon_1 FROM users AS users_1 JOIN addresses AS addresses_1 ON users_1.id = addresses_1.user_id'), (ua.id + aa.id, (ua, aa.user), 'SELECT users_1.id + addresses_1.id AS anon_1 FROM addresses AS addresses_1 JOIN users AS users_1 ON users_1.id = addresses_1.user_id')]:
            q = s.query(crit)
            mzero = q._compile_state()._entity_zero()
            is_(mzero, q._compile_state()._entities[0].entity_zero)
            q = q.join(*j)
            self.assert_compile(q, exp)

    def test_aliased_adapt_on_names(self):
        if False:
            return 10
        (User, Address) = (self.classes.User, self.classes.Address)
        sess = fixture_session()
        agg_address = sess.query(Address.id, func.sum(func.length(Address.email_address)).label('email_address')).group_by(Address.user_id)
        ag1 = aliased(Address, agg_address.subquery())
        ag2 = aliased(Address, agg_address.subquery(), adapt_on_names=True)
        self.assert_compile(sess.query(User, ag1.email_address).join(ag1, User.addresses).filter(ag1.email_address > 5), 'SELECT users.id AS users_id, users.name AS users_name, addresses.email_address AS addresses_email_address FROM users JOIN (SELECT addresses.id AS id, sum(length(addresses.email_address)) AS email_address FROM addresses GROUP BY addresses.user_id) AS anon_1 ON users.id = addresses.user_id, addresses WHERE addresses.email_address > :email_address_1')
        self.assert_compile(sess.query(User, ag2.email_address).join(ag2, User.addresses).filter(ag2.email_address > 5), 'SELECT users.id AS users_id, users.name AS users_name, anon_1.email_address AS anon_1_email_address FROM users JOIN (SELECT addresses.id AS id, sum(length(addresses.email_address)) AS email_address FROM addresses GROUP BY addresses.user_id) AS anon_1 ON users.id = addresses.user_id WHERE anon_1.email_address > :email_address_1')

class SelectFromTest(QueryTest, AssertsCompiledSQL):
    run_setup_mappers = None
    __dialect__ = 'default'

    def test_replace_with_select(self):
        if False:
            print('Hello World!')
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sel = users.select().where(users.c.id.in_([7, 8])).alias()
        sess = fixture_session()
        ua = aliased(User, sel)
        eq_(sess.query(ua).all(), [User(id=7), User(id=8)])
        eq_(sess.query(ua).filter(ua.id == 8).all(), [User(id=8)])
        eq_(sess.query(ua).order_by(desc(ua.name)).all(), [User(name='jack', id=7), User(name='ed', id=8)])
        eq_(sess.query(ua).order_by(asc(ua.name)).all(), [User(name='ed', id=8), User(name='jack', id=7)])
        eq_(sess.query(ua).options(joinedload(ua.addresses)).first(), User(name='jack', addresses=[Address(id=1)]))

    def test_select_from_aliased_one(self):
        if False:
            while True:
                i = 10
        (User, users) = (self.classes.User, self.tables.users)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        not_users = table('users', column('id'), column('name'))
        ua = aliased(User, select(not_users).alias(), adapt_on_names=True)
        q = sess.query(ua.name).order_by(ua.name)
        self.assert_compile(q, 'SELECT anon_1.name AS anon_1_name FROM (SELECT users.id AS id, users.name AS name FROM users) AS anon_1 ORDER BY anon_1.name')
        eq_(q.all(), [('chuck',), ('ed',), ('fred',), ('jack',)])

    def test_select_from_aliased_two(self):
        if False:
            for i in range(10):
                print('nop')
        (User, users) = (self.classes.User, self.tables.users)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        ua = aliased(User)
        q = sess.query(ua.name).order_by(ua.name)
        self.assert_compile(q, 'SELECT users_1.name AS users_1_name FROM users AS users_1 ORDER BY users_1.name')
        eq_(q.all(), [('chuck',), ('ed',), ('fred',), ('jack',)])

    def test_differentiate_self_external(self):
        if False:
            print('Hello World!')
        'test some different combinations of joining a table to a subquery of\n        itself.'
        (users, User) = (self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(User, users)
        sess = fixture_session()
        sel = sess.query(User).filter(User.id.in_([7, 8])).subquery()
        ualias = aliased(User)
        ua = aliased(User, sel)
        self.assert_compile(sess.query(User).join(sel, User.id > sel.c.id), 'SELECT users.id AS users_id, users.name AS users_name FROM users JOIN (SELECT users.id AS id, users.name AS name FROM users WHERE users.id IN (__[POSTCOMPILE_id_1])) AS anon_1 ON users.id > anon_1.id')
        self.assert_compile(sess.query(ualias).select_from(ua).filter(ualias.id > ua.id), 'SELECT users_1.id AS users_1_id, users_1.name AS users_1_name FROM (SELECT users.id AS id, users.name AS name FROM users WHERE users.id IN (__[POSTCOMPILE_id_1])) AS anon_1, users AS users_1 WHERE users_1.id > anon_1.id', check_post_param={'id_1': [7, 8]})
        self.assert_compile(sess.query(ualias).select_from(ua).join(ualias, ualias.id > ua.id), 'SELECT users_1.id AS users_1_id, users_1.name AS users_1_name FROM (SELECT users.id AS id, users.name AS name FROM users WHERE users.id IN (__[POSTCOMPILE_id_1])) AS anon_1 JOIN users AS users_1 ON users_1.id > anon_1.id', check_post_param={'id_1': [7, 8]})
        self.assert_compile(sess.query(ualias).select_from(ua).join(ualias, ualias.id > ua.id), 'SELECT users_1.id AS users_1_id, users_1.name AS users_1_name FROM (SELECT users.id AS id, users.name AS name FROM users WHERE users.id IN (__[POSTCOMPILE_id_1])) AS anon_1 JOIN users AS users_1 ON users_1.id > anon_1.id', check_post_param={'id_1': [7, 8]})
        salias = aliased(User, sel)
        self.assert_compile(sess.query(salias).join(ualias, ualias.id > salias.id), 'SELECT anon_1.id AS anon_1_id, anon_1.name AS anon_1_name FROM (SELECT users.id AS id, users.name AS name FROM users WHERE users.id IN (__[POSTCOMPILE_id_1])) AS anon_1 JOIN users AS users_1 ON users_1.id > anon_1.id', check_post_param={'id_1': [7, 8]})

    def test_aliased_class_vs_nonaliased(self):
        if False:
            for i in range(10):
                print('nop')
        (User, users) = (self.classes.User, self.tables.users)
        self.mapper_registry.map_imperatively(User, users)
        ua = aliased(User)
        sess = fixture_session()
        self.assert_compile(sess.query(User).select_from(ua).join(User, ua.name > User.name), 'SELECT users.id AS users_id, users.name AS users_name FROM users AS users_1 JOIN users ON users_1.name > users.name')
        self.assert_compile(sess.query(User.name).select_from(ua).join(User, ua.name > User.name), 'SELECT users.name AS users_name FROM users AS users_1 JOIN users ON users_1.name > users.name')
        self.assert_compile(sess.query(ua.name).select_from(ua).join(User, ua.name > User.name), 'SELECT users_1.name AS users_1_name FROM users AS users_1 JOIN users ON users_1.name > users.name')
        self.assert_compile(sess.query(ua).select_from(User).join(ua, ua.name > User.name), 'SELECT users_1.id AS users_1_id, users_1.name AS users_1_name FROM users JOIN users AS users_1 ON users_1.name > users.name')
        self.assert_compile(sess.query(ua).select_from(User).join(ua, User.name > ua.name), 'SELECT users_1.id AS users_1_id, users_1.name AS users_1_name FROM users JOIN users AS users_1 ON users.name > users_1.name')

    def test_join_relname_from_selected_from(self):
        if False:
            return 10
        (User, Address) = (self.classes.User, self.classes.Address)
        (users, addresses) = (self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(self.mapper_registry.map_imperatively(Address, addresses), backref='user')})
        sess = fixture_session()
        self.assert_compile(sess.query(User).select_from(Address).join(Address.user), 'SELECT users.id AS users_id, users.name AS users_name FROM addresses JOIN users ON users.id = addresses.user_id')

    def test_filter_by_selected_from(self):
        if False:
            for i in range(10):
                print('nop')
        (User, Address) = (self.classes.User, self.classes.Address)
        (users, addresses) = (self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(self.mapper_registry.map_imperatively(Address, addresses))})
        sess = fixture_session()
        self.assert_compile(sess.query(User).select_from(Address).filter_by(email_address='ed').join(User), 'SELECT users.id AS users_id, users.name AS users_name FROM addresses JOIN users ON users.id = addresses.user_id WHERE addresses.email_address = :email_address_1')

    def test_join_ent_selected_from(self):
        if False:
            for i in range(10):
                print('nop')
        (User, Address) = (self.classes.User, self.classes.Address)
        (users, addresses) = (self.tables.users, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(self.mapper_registry.map_imperatively(Address, addresses))})
        sess = fixture_session()
        self.assert_compile(sess.query(User).select_from(Address).join(User), 'SELECT users.id AS users_id, users.name AS users_name FROM addresses JOIN users ON users.id = addresses.user_id')

    def test_join(self):
        if False:
            return 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sel = users.select().where(users.c.id.in_([7, 8]))
        sess = fixture_session()
        ua = aliased(User, sel.subquery())
        eq_(sess.query(ua).join(ua.addresses).add_entity(Address).order_by(ua.id).order_by(Address.id).all(), [(User(name='jack', id=7), Address(user_id=7, email_address='jack@bean.com', id=1)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@wood.com', id=2)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@bettyboop.com', id=3)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@lala.com', id=4))])
        adalias = aliased(Address)
        ua = aliased(User, sel.subquery())
        eq_(sess.query(ua).join(ua.addresses.of_type(adalias)).add_entity(adalias).order_by(ua.id).order_by(adalias.id).all(), [(User(name='jack', id=7), Address(user_id=7, email_address='jack@bean.com', id=1)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@wood.com', id=2)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@bettyboop.com', id=3)), (User(name='ed', id=8), Address(user_id=8, email_address='ed@lala.com', id=4))])

    def test_more_joins(self):
        if False:
            print('Hello World!')
        (users, Keyword, orders, items, order_items, Order, Item, User, keywords, item_keywords) = (self.tables.users, self.classes.Keyword, self.tables.orders, self.tables.items, self.tables.order_items, self.classes.Order, self.classes.Item, self.classes.User, self.tables.keywords, self.tables.item_keywords)
        self.mapper_registry.map_imperatively(User, users, properties={'orders': relationship(Order, backref='user')})
        self.mapper_registry.map_imperatively(Order, orders, properties={'items': relationship(Item, secondary=order_items, order_by=items.c.id)})
        self.mapper_registry.map_imperatively(Item, items, properties={'keywords': relationship(Keyword, secondary=item_keywords, order_by=keywords.c.id)})
        self.mapper_registry.map_imperatively(Keyword, keywords)
        sess = fixture_session()
        sel = users.select().where(users.c.id.in_([7, 8]))
        ua = aliased(User, sel.subquery())
        eq_(sess.query(ua).join(ua.orders).join(Order.items).join(Item.keywords).filter(Keyword.name.in_(['red', 'big', 'round'])).all(), [User(name='jack', id=7)])

    def test_very_nested_joins_with_joinedload(self):
        if False:
            print('Hello World!')
        (users, Keyword, orders, items, order_items, Order, Item, User, keywords, item_keywords) = (self.tables.users, self.classes.Keyword, self.tables.orders, self.tables.items, self.tables.order_items, self.classes.Order, self.classes.Item, self.classes.User, self.tables.keywords, self.tables.item_keywords)
        self.mapper_registry.map_imperatively(User, users, properties={'orders': relationship(Order, backref='user')})
        self.mapper_registry.map_imperatively(Order, orders, properties={'items': relationship(Item, secondary=order_items, order_by=items.c.id)})
        self.mapper_registry.map_imperatively(Item, items, properties={'keywords': relationship(Keyword, secondary=item_keywords, order_by=keywords.c.id)})
        self.mapper_registry.map_imperatively(Keyword, keywords)
        sess = fixture_session()
        sel = users.select().where(users.c.id.in_([7, 8]))
        ua = aliased(User, sel.subquery())

        def go():
            if False:
                for i in range(10):
                    print('nop')
            eq_(sess.query(ua).options(joinedload(ua.orders).joinedload(Order.items).joinedload(Item.keywords)).join(ua.orders).join(Order.items).join(Item.keywords).filter(Keyword.name.in_(['red', 'big', 'round'])).all(), [User(name='jack', orders=[Order(description='order 1', items=[Item(description='item 1', keywords=[Keyword(name='red'), Keyword(name='big'), Keyword(name='round')]), Item(description='item 2', keywords=[Keyword(name='red', id=2), Keyword(name='small', id=5), Keyword(name='square')]), Item(description='item 3', keywords=[Keyword(name='green', id=3), Keyword(name='big', id=4), Keyword(name='round', id=6)])]), Order(description='order 3', items=[Item(description='item 3', keywords=[Keyword(name='green', id=3), Keyword(name='big', id=4), Keyword(name='round', id=6)]), Item(description='item 4', keywords=[], id=4), Item(description='item 5', keywords=[], id=5)]), Order(description='order 5', items=[Item(description='item 5', keywords=[])])])])
        self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()
        sel2 = orders.select().where(orders.c.id.in_([1, 2, 3]))
        oa = aliased(Order, sel2.subquery())
        eq_(sess.query(oa).join(oa.items).join(Item.keywords).filter(Keyword.name == 'red').order_by(oa.id).all(), [Order(description='order 1', id=1), Order(description='order 2', id=2)])

    def test_replace_with_eager(self):
        if False:
            while True:
                i = 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, order_by=addresses.c.id)})
        self.mapper_registry.map_imperatively(Address, addresses)
        sel = users.select().where(users.c.id.in_([7, 8]))
        sess = fixture_session()
        ua = aliased(User, sel.subquery())

        def go():
            if False:
                i = 10
                return i + 15
            eq_(sess.query(ua).options(joinedload(ua.addresses)).order_by(ua.id).all(), [User(id=7, addresses=[Address(id=1)]), User(id=8, addresses=[Address(id=2), Address(id=3), Address(id=4)])])
        self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()

        def go():
            if False:
                for i in range(10):
                    print('nop')
            eq_(sess.query(ua).options(joinedload(ua.addresses)).filter(ua.id == 8).order_by(ua.id).all(), [User(id=8, addresses=[Address(id=2), Address(id=3), Address(id=4)])])
        self.assert_sql_count(testing.db, go, 1)
        sess.expunge_all()

        def go():
            if False:
                for i in range(10):
                    print('nop')
            eq_(sess.query(ua).options(joinedload(ua.addresses)).order_by(ua.id).offset(1).limit(1).scalar(), User(id=8, addresses=[Address(id=2), Address(id=3), Address(id=4)]))
        self.assert_sql_count(testing.db, go, 1)

class CustomJoinTest(QueryTest):
    run_setup_mappers = None

    def test_double_same_mappers_explicit_alias(self):
        if False:
            for i in range(10):
                print('nop')
        'test aliasing of joins with a custom join condition'
        (addresses, items, order_items, orders, Item, User, Address, Order, users) = (self.tables.addresses, self.tables.items, self.tables.order_items, self.tables.orders, self.classes.Item, self.classes.User, self.classes.Address, self.classes.Order, self.tables.users)
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(Order, orders, properties={'items': relationship(Item, secondary=order_items, lazy='select', order_by=items.c.id)})
        self.mapper_registry.map_imperatively(Item, items)
        self.mapper_registry.map_imperatively(User, users, properties=dict(addresses=relationship(Address, lazy='select'), open_orders=relationship(Order, primaryjoin=and_(orders.c.isopen == 1, users.c.id == orders.c.user_id), lazy='select', viewonly=True), closed_orders=relationship(Order, primaryjoin=and_(orders.c.isopen == 0, users.c.id == orders.c.user_id), lazy='select', viewonly=True)))
        q = fixture_session().query(User)
        oo = aliased(Order)
        co = aliased(Order)
        oi = aliased(Item)
        ci = aliased(Item)
        eq_(q.join(oo, User.open_orders).join(oi, oo.items).filter(oi.id == 4).join(User.closed_orders.of_type(co)).join(co.items.of_type(ci)).filter(ci.id == 3).all(), [User(id=7)])

class ExternalColumnsTest(QueryTest):
    """test mappers with SQL-expressions added as column properties."""
    run_setup_mappers = None

    def test_external_columns_bad(self):
        if False:
            print('Hello World!')
        (users, User) = (self.tables.users, self.classes.User)
        assert_raises_message(sa_exc.ArgumentError, "not represented in the mapper's table", self.mapper_registry.map_imperatively, User, users, properties={'concat': users.c.id * 2})
        clear_mappers()

    def test_external_columns(self):
        if False:
            for i in range(10):
                print('nop')
        'test querying mappings that reference external columns or\n        selectables.'
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'concat': column_property(users.c.id * 2), 'count': column_property(select(func.count(addresses.c.id)).where(users.c.id == addresses.c.user_id).correlate(users).scalar_subquery())})
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User)})
        sess = fixture_session()
        sess.query(Address).options(joinedload(Address.user)).all()
        eq_(sess.query(User).all(), [User(id=7, concat=14, count=1), User(id=8, concat=16, count=3), User(id=9, concat=18, count=1), User(id=10, concat=20, count=0)])
        address_result = [Address(id=1, user=User(id=7, concat=14, count=1)), Address(id=2, user=User(id=8, concat=16, count=3)), Address(id=3, user=User(id=8, concat=16, count=3)), Address(id=4, user=User(id=8, concat=16, count=3)), Address(id=5, user=User(id=9, concat=18, count=1))]
        eq_(sess.query(Address).all(), address_result)
        for x in range(2):
            sess.expunge_all()

            def go():
                if False:
                    while True:
                        i = 10
                eq_(sess.query(Address).options(joinedload(Address.user)).order_by(Address.id).all(), address_result)
            self.assert_sql_count(testing.db, go, 1)
        ualias = aliased(User)
        eq_(sess.query(Address, ualias).join(ualias, Address.user).all(), [(address, address.user) for address in address_result])
        ualias2 = aliased(User)
        eq_(sess.query(Address, ualias.count).join(ualias, Address.user).join(ualias2, Address.user).order_by(Address.id).all(), [(Address(id=1), 1), (Address(id=2), 3), (Address(id=3), 3), (Address(id=4), 3), (Address(id=5), 1)])
        eq_(sess.query(Address, ualias.concat, ualias.count).join(Address.user.of_type(ualias)).join(Address.user.of_type(ualias2)).order_by(Address.id).all(), [(Address(id=1), 14, 1), (Address(id=2), 16, 3), (Address(id=3), 16, 3), (Address(id=4), 16, 3), (Address(id=5), 18, 1)])
        ua = aliased(User)
        eq_(sess.query(Address, ua.concat, ua.count).join(Address.user.of_type(ua)).options(joinedload(Address.user)).order_by(Address.id).all(), [(Address(id=1, user=User(id=7, concat=14, count=1)), 14, 1), (Address(id=2, user=User(id=8, concat=16, count=3)), 16, 3), (Address(id=3, user=User(id=8, concat=16, count=3)), 16, 3), (Address(id=4, user=User(id=8, concat=16, count=3)), 16, 3), (Address(id=5, user=User(id=9, concat=18, count=1)), 18, 1)])
        eq_(list(sess.query(Address).join(Address.user).with_entities(Address.id, User.id, User.concat, User.count)), [(1, 7, 14, 1), (2, 8, 16, 3), (3, 8, 16, 3), (4, 8, 16, 3), (5, 9, 18, 1)])
        eq_(list(sess.query(Address, ua).join(Address.user.of_type(ua)).with_entities(Address.id, ua.id, ua.concat, ua.count)), [(1, 7, 14, 1), (2, 8, 16, 3), (3, 8, 16, 3), (4, 8, 16, 3), (5, 9, 18, 1)])

    def test_external_columns_joinedload(self):
        if False:
            print('Hello World!')
        (users, orders, User, Address, Order, addresses) = (self.tables.users, self.tables.orders, self.classes.User, self.classes.Address, self.classes.Order, self.tables.addresses)
        self.mapper_registry.map_imperatively(User, users, properties={'addresses': relationship(Address, backref='user', order_by=addresses.c.id), 'concat': column_property(users.c.id * 2), 'count': column_property(select(func.count(addresses.c.id)).where(users.c.id == addresses.c.user_id).correlate(users).scalar_subquery())})
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(Order, orders, properties={'address': relationship(Address)})
        configure_mappers()
        sess = fixture_session()

        def go():
            if False:
                return 10
            o1 = sess.get(Order, 1, options=[joinedload(Order.address).joinedload(Address.user)])
            eq_(o1.address.user.count, 1)
        self.assert_sql_count(testing.db, go, 1)
        sess = fixture_session()

        def go():
            if False:
                return 10
            o1 = sess.query(Order).options(joinedload(Order.address).joinedload(Address.user)).first()
            eq_(o1.address.user.count, 1)
        self.assert_sql_count(testing.db, go, 1)

    def test_external_columns_compound(self):
        if False:
            while True:
                i = 10
        (users, Address, addresses, User) = (self.tables.users, self.classes.Address, self.tables.addresses, self.classes.User)
        self.mapper_registry.map_imperatively(User, users, properties={'fullname': column_property(users.c.name.label('x'))})
        self.mapper_registry.map_imperatively(Address, addresses, properties={'username': column_property(select(User.fullname).where(User.id == addresses.c.user_id).label('y'))})
        sess = fixture_session()
        a1 = sess.query(Address).first()
        eq_(a1.username, 'jack')
        sess = fixture_session()
        subq = sess.query(Address).subquery()
        aa = aliased(Address, subq)
        a1 = sess.query(aa).first()
        eq_(a1.username, 'jack')

class TestOverlyEagerEquivalentCols(fixtures.MappedTest):

    @classmethod
    def define_tables(cls, metadata):
        if False:
            return 10
        Table('base', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('data', String(50)))
        Table('sub1', metadata, Column('id', Integer, ForeignKey('base.id'), primary_key=True), Column('data', String(50)))
        Table('sub2', metadata, Column('id', Integer, ForeignKey('base.id'), ForeignKey('sub1.id'), primary_key=True), Column('data', String(50)))

    def test_equivs(self):
        if False:
            return 10
        (base, sub2, sub1) = (self.tables.base, self.tables.sub2, self.tables.sub1)

        class Base(ComparableEntity):
            pass

        class Sub1(ComparableEntity):
            pass

        class Sub2(ComparableEntity):
            pass
        self.mapper_registry.map_imperatively(Base, base, properties={'sub1': relationship(Sub1), 'sub2': relationship(Sub2)})
        self.mapper_registry.map_imperatively(Sub1, sub1)
        self.mapper_registry.map_imperatively(Sub2, sub2)
        sess = fixture_session()
        s11 = Sub1(data='s11')
        s12 = Sub1(data='s12')
        b1 = Base(data='b1', sub1=[s11], sub2=[])
        b2 = Base(data='b1', sub1=[s12], sub2=[])
        sess.add(b1)
        sess.add(b2)
        sess.flush()

class LabelCollideTest(fixtures.MappedTest):
    """Test handling for a label collision.  This collision
    is handled by core, see ticket:2702 as well as
    test/sql/test_selectable->WithLabelsTest.  here we want
    to make sure the end result is as we expect.

    """

    @classmethod
    def define_tables(cls, metadata):
        if False:
            while True:
                i = 10
        Table('foo', metadata, Column('id', Integer, primary_key=True), Column('bar_id', Integer))
        Table('foo_bar', metadata, Column('id', Integer, primary_key=True))

    @classmethod
    def setup_classes(cls):
        if False:
            while True:
                i = 10

        class Foo(cls.Basic):
            pass

        class Bar(cls.Basic):
            pass

    @classmethod
    def setup_mappers(cls):
        if False:
            i = 10
            return i + 15
        cls.mapper_registry.map_imperatively(cls.classes.Foo, cls.tables.foo)
        cls.mapper_registry.map_imperatively(cls.classes.Bar, cls.tables.foo_bar)

    @classmethod
    def insert_data(cls, connection):
        if False:
            while True:
                i = 10
        s = Session(connection)
        s.add_all([cls.classes.Foo(id=1, bar_id=2), cls.classes.Bar(id=3)])
        s.commit()

    def test_overlap_plain(self):
        if False:
            i = 10
            return i + 15
        s = fixture_session()
        row = s.query(self.classes.Foo, self.classes.Bar).join(self.classes.Bar, true()).all()[0]

        def go():
            if False:
                return 10
            eq_(row.Foo.id, 1)
            eq_(row.Foo.bar_id, 2)
            eq_(row.Bar.id, 3)
        self.assert_sql_count(testing.db, go, 0)

    def test_overlap_subquery(self):
        if False:
            print('Hello World!')
        s = fixture_session()
        subq = s.query(self.classes.Foo, self.classes.Bar).join(self.classes.Bar, true()).subquery()
        fa = aliased(self.classes.Foo, subq, name='Foo')
        ba = aliased(self.classes.Bar, subq, name='Bar')
        row = s.query(fa, ba).all()[0]

        def go():
            if False:
                print('Hello World!')
            eq_(row.Foo.id, 1)
            eq_(row.Foo.bar_id, 2)
            eq_(row.Bar.id, 3)
        self.assert_sql_count(testing.db, go, 0)

class CorrelateORMTest(fixtures.TestBase, testing.AssertsCompiledSQL):
    __dialect__ = 'default'

    @testing.fixture
    def mapping(self):
        if False:
            for i in range(10):
                print('nop')
        Base = declarative_base()

        def go(include_property, correlate_style, include_from):
            if False:
                i = 10
                return i + 15

            class Address(Base):
                __tablename__ = 'addresses'
                id = Column(Integer, primary_key=True)
                user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
                city = Column(Text)

            class User(Base):
                __tablename__ = 'users'
                id = Column(Integer, primary_key=True)
                name = Column(Text)
            stmt = select(func.count(Address.id)).where(Address.user_id == User.id)
            if include_from:
                stmt = stmt.select_from(Address)
            if include_property:
                if correlate_style == 'correlate':
                    User.total_addresses = column_property(stmt.correlate(User).scalar_subquery())
                elif correlate_style == 'correlate_except':
                    User.total_addresses = column_property(stmt.correlate_except(Address).scalar_subquery())
                elif correlate_style is None:
                    User.total_addresses = column_property(stmt.scalar_subquery())
                total_addresses = None
            else:

                def total_addresses(cls):
                    if False:
                        while True:
                            i = 10
                    stmt = select(func.count(Address.id)).where(Address.user_id == cls.id)
                    if correlate_style == 'correlate':
                        stmt = stmt.correlate(cls)
                    elif correlate_style == 'correlate_except':
                        stmt = stmt.correlate_except(Address)
                    stmt = stmt.scalar_subquery()
                    return stmt
            return (User, Address, total_addresses)
        yield go
        Base.registry.dispose()

    def _combinations(fn):
        if False:
            for i in range(10):
                print('nop')
        return testing.combinations((True,), (False,), argnames='include_property')(testing.combinations(('correlate',), ('correlate_except',), (None,), argnames='correlate_style')(testing.combinations((True,), False, argnames='include_from')(fn)))

    @_combinations
    def test_correlate_to_cte_legacy(self, mapping, include_property, correlate_style, include_from):
        if False:
            print('Hello World!')
        (User, Address, total_addresses) = mapping(include_property, correlate_style, include_from)
        session = fixture_session()
        filtered_users = session.query(User.id, User.name).join(Address).filter(Address.city == 'somewhere').cte('filtered_users')
        filtered_users_alias = aliased(User, filtered_users)
        paginated_users = session.query(filtered_users_alias.id, filtered_users_alias.name).order_by(func.lower(filtered_users_alias.name).asc()).limit(25).cte('paginated_users')
        paginated_users_alias = aliased(User, paginated_users)
        if total_addresses:
            q = session.query(paginated_users_alias, total_addresses(paginated_users_alias))
        else:
            q = session.query(paginated_users_alias)
        self.assert_compile(q, 'WITH filtered_users AS (SELECT users.id AS id, users.name AS name FROM users JOIN addresses ON users.id = addresses.user_id WHERE addresses.city = :city_1), paginated_users AS (SELECT filtered_users.id AS id, filtered_users.name AS name FROM filtered_users ORDER BY lower(filtered_users.name) ASC LIMIT :param_1) SELECT paginated_users.id AS paginated_users_id, paginated_users.name AS paginated_users_name, (SELECT count(addresses.id) AS count_1 FROM addresses WHERE addresses.user_id = paginated_users.id) AS anon_1 FROM paginated_users')
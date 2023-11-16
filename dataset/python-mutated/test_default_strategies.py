import sqlalchemy as sa
from sqlalchemy import testing
from sqlalchemy import util
from sqlalchemy.orm import defaultload
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import relationship
from sqlalchemy.orm import subqueryload
from sqlalchemy.testing import eq_
from sqlalchemy.testing.assertions import expect_raises_message
from sqlalchemy.testing.fixtures import fixture_session
from test.orm import _fixtures

class DefaultStrategyOptionsTest(_fixtures.FixtureTest):

    def _assert_fully_loaded(self, users):
        if False:
            for i in range(10):
                print('nop')

        def go():
            if False:
                while True:
                    i = 10
            eq_(users, self.static.user_all_result)
            f = util.flatten_iterator
            assert any([i.keywords for i in f([o.items for o in f([u.orders for u in users])])])
        self.assert_sql_count(testing.db, go, 0)

    def _assert_addresses_loaded(self, users):
        if False:
            print('Hello World!')

        def go():
            if False:
                print('Hello World!')
            for (u, static) in zip(users, self.static.user_all_result):
                eq_(u.addresses, static.addresses)
        self.assert_sql_count(testing.db, go, 0)

    def _downgrade_fixture(self):
        if False:
            for i in range(10):
                print('nop')
        (users, Keyword, items, order_items, orders, Item, User, Address, keywords, item_keywords, Order, addresses) = (self.tables.users, self.classes.Keyword, self.tables.items, self.tables.order_items, self.tables.orders, self.classes.Item, self.classes.User, self.classes.Address, self.tables.keywords, self.tables.item_keywords, self.classes.Order, self.tables.addresses)
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(Keyword, keywords)
        self.mapper_registry.map_imperatively(Item, items, properties=dict(keywords=relationship(Keyword, secondary=item_keywords, lazy='subquery', order_by=item_keywords.c.keyword_id)))
        self.mapper_registry.map_imperatively(Order, orders, properties=dict(items=relationship(Item, secondary=order_items, lazy='subquery', order_by=order_items.c.item_id)))
        self.mapper_registry.map_imperatively(User, users, properties=dict(addresses=relationship(Address, lazy='joined', order_by=addresses.c.id), orders=relationship(Order, lazy='joined', order_by=orders.c.id)))
        return fixture_session()

    def _upgrade_fixture(self):
        if False:
            print('Hello World!')
        (users, Keyword, items, order_items, orders, Item, User, Address, keywords, item_keywords, Order, addresses) = (self.tables.users, self.classes.Keyword, self.tables.items, self.tables.order_items, self.tables.orders, self.classes.Item, self.classes.User, self.classes.Address, self.tables.keywords, self.tables.item_keywords, self.classes.Order, self.tables.addresses)
        self.mapper_registry.map_imperatively(Address, addresses)
        self.mapper_registry.map_imperatively(Keyword, keywords)
        self.mapper_registry.map_imperatively(Item, items, properties=dict(keywords=relationship(Keyword, secondary=item_keywords, lazy='select', order_by=item_keywords.c.keyword_id)))
        self.mapper_registry.map_imperatively(Order, orders, properties=dict(items=relationship(Item, secondary=order_items, lazy=True, order_by=order_items.c.item_id)))
        self.mapper_registry.map_imperatively(User, users, properties=dict(addresses=relationship(Address, lazy=True, order_by=addresses.c.id), orders=relationship(Order, order_by=orders.c.id)))
        return fixture_session()

    def test_downgrade_baseline(self):
        if False:
            i = 10
            return i + 15
        'Mapper strategy defaults load as expected\n        (compare to rest of DefaultStrategyOptionsTest downgrade tests).'
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                i = 10
                return i + 15
            users[:] = sess.query(self.classes.User).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 3)
        self._assert_fully_loaded(users)

    def test_disable_eagerloads(self):
        if False:
            print('Hello World!')
        'Mapper eager load strategy defaults can be shut off\n        with enable_eagerloads(False).'
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                return 10
            users[:] = sess.query(self.classes.User).enable_eagerloads(False).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)

        def go():
            if False:
                i = 10
                return i + 15
            users[0].orders
        self.assert_sql_count(testing.db, go, 3)

    def test_last_one_wins(self):
        if False:
            while True:
                i = 10
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[:] = sess.query(self.classes.User).options(subqueryload('*')).options(joinedload(self.classes.User.addresses)).options(sa.orm.lazyload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)
        self._assert_addresses_loaded(users)

    def test_star_must_be_alone(self):
        if False:
            while True:
                i = 10
        self._downgrade_fixture()
        User = self.classes.User
        with expect_raises_message(sa.exc.ArgumentError, 'Wildcard token cannot be followed by another entity'):
            subqueryload('*', User.addresses)

    def test_star_cant_be_followed(self):
        if False:
            while True:
                i = 10
        self._downgrade_fixture()
        User = self.classes.User
        Order = self.classes.Order
        with expect_raises_message(sa.exc.ArgumentError, 'Wildcard token cannot be followed by another entity'):
            subqueryload(User.addresses).joinedload('*').selectinload(Order.items)

    def test_global_star_ignored_no_entities_unbound(self):
        if False:
            i = 10
            return i + 15
        sess = self._downgrade_fixture()
        User = self.classes.User
        opt = sa.orm.lazyload('*')
        q = sess.query(User.name).options(opt)
        eq_(q.all(), [('jack',), ('ed',), ('fred',), ('chuck',)])

    def test_global_star_ignored_no_entities_bound(self):
        if False:
            i = 10
            return i + 15
        sess = self._downgrade_fixture()
        User = self.classes.User
        opt = sa.orm.Load(User).lazyload('*')
        q = sess.query(User.name).options(opt)
        eq_(q.all(), [('jack',), ('ed',), ('fred',), ('chuck',)])

    def test_select_with_joinedload(self):
        if False:
            return 10
        "Mapper load strategy defaults can be downgraded with\n        lazyload('*') option, while explicit joinedload() option\n        is still honored"
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                print('Hello World!')
            users[:] = sess.query(self.classes.User).options(sa.orm.lazyload('*')).options(joinedload(self.classes.User.addresses)).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)
        self._assert_addresses_loaded(users)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[0].orders
        self.assert_sql_count(testing.db, go, 3)

    def test_select_with_subqueryload(self):
        if False:
            return 10
        "Mapper load strategy defaults can be downgraded with\n        lazyload('*') option, while explicit subqueryload() option\n        is still honored"
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[:] = sess.query(self.classes.User).options(sa.orm.lazyload('*')).options(subqueryload(self.classes.User.orders)).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 2)

        def go():
            if False:
                return 10
            for (u, static) in zip(users, self.static.user_all_result):
                assert len(u.orders) == len(static.orders)
        self.assert_sql_count(testing.db, go, 0)

        def go():
            if False:
                return 10
            for i in users[0].orders[0].items:
                i.keywords
        self.assert_sql_count(testing.db, go, 2)
        eq_(users, self.static.user_all_result)

    def test_noload_with_joinedload(self):
        if False:
            while True:
                i = 10
        "Mapper load strategy defaults can be downgraded with\n        noload('*') option, while explicit joinedload() option\n        is still honored"
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                print('Hello World!')
            users[:] = sess.query(self.classes.User).options(sa.orm.noload('*')).options(joinedload(self.classes.User.addresses)).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)
        self._assert_addresses_loaded(users)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            for u in users:
                assert u.orders == []
        self.assert_sql_count(testing.db, go, 0)

    def test_noload_with_subqueryload(self):
        if False:
            print('Hello World!')
        "Mapper load strategy defaults can be downgraded with\n        noload('*') option, while explicit subqueryload() option\n        is still honored"
        sess = self._downgrade_fixture()
        users = []

        def go():
            if False:
                print('Hello World!')
            users[:] = sess.query(self.classes.User).options(sa.orm.noload('*')).options(subqueryload(self.classes.User.orders)).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 2)

        def go():
            if False:
                print('Hello World!')
            for (u, static) in zip(users, self.static.user_all_result):
                assert len(u.orders) == len(static.orders)
            for u in users:
                for o in u.orders:
                    assert o.items == []
        self.assert_sql_count(testing.db, go, 0)

    def test_joined(self):
        if False:
            return 10
        "Mapper load strategy defaults can be upgraded with\n        joinedload('*') option."
        sess = self._upgrade_fixture()
        users = []

        def go():
            if False:
                while True:
                    i = 10
            users[:] = sess.query(self.classes.User).options(joinedload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)
        self._assert_fully_loaded(users)

    def test_joined_path_wildcards(self):
        if False:
            print('Hello World!')
        sess = self._upgrade_fixture()
        users = []
        (User, Order, Item) = self.classes('User', 'Order', 'Item')

        def go():
            if False:
                i = 10
                return i + 15
            users[:] = sess.query(User).options(joinedload('*')).options(defaultload(User.addresses).joinedload('*')).options(defaultload(User.orders).joinedload('*')).options(defaultload(User.orders).defaultload(Order.items).joinedload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 1)
        self._assert_fully_loaded(users)

    def test_joined_with_lazyload(self):
        if False:
            while True:
                i = 10
        "Mapper load strategy defaults can be upgraded with\n        joinedload('*') option, while explicit lazyload() option\n        is still honored"
        sess = self._upgrade_fixture()
        users = []
        (User, Order, Item) = self.classes('User', 'Order', 'Item')

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[:] = sess.query(User).options(defaultload(User.orders).defaultload(Order.items).lazyload(Item.keywords)).options(joinedload('*')).order_by(User.id).all()
        self.assert_sql_count(testing.db, go, 1)

        def go():
            if False:
                print('Hello World!')
            eq_(users, self.static.user_all_result)
        self.assert_sql_count(testing.db, go, 0)

        def go():
            if False:
                return 10
            users[0].orders[0].items[0]
        self.assert_sql_count(testing.db, go, 0)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[0].orders[0].items[0].keywords
        self.assert_sql_count(testing.db, go, 1)

    def test_joined_with_subqueryload(self):
        if False:
            for i in range(10):
                print('nop')
        "Mapper load strategy defaults can be upgraded with\n        joinedload('*') option, while explicit subqueryload() option\n        is still honored"
        sess = self._upgrade_fixture()
        users = []

        def go():
            if False:
                return 10
            users[:] = sess.query(self.classes.User).options(subqueryload(self.classes.User.addresses)).options(joinedload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 2)
        self._assert_fully_loaded(users)

    def test_subquery(self):
        if False:
            i = 10
            return i + 15
        "Mapper load strategy defaults can be upgraded with\n        subqueryload('*') option."
        sess = self._upgrade_fixture()
        users = []

        def go():
            if False:
                return 10
            users[:] = sess.query(self.classes.User).options(subqueryload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 5)
        self._assert_fully_loaded(users)

    def test_subquery_path_wildcards(self):
        if False:
            return 10
        sess = self._upgrade_fixture()
        users = []
        (User, Order) = self.classes('User', 'Order')

        def go():
            if False:
                return 10
            users[:] = sess.query(User).options(subqueryload('*')).options(defaultload(User.addresses).subqueryload('*')).options(defaultload(User.orders).subqueryload('*')).options(defaultload(User.orders).defaultload(Order.items).subqueryload('*')).order_by(User.id).all()
        self.assert_sql_count(testing.db, go, 5)
        self._assert_fully_loaded(users)

    def test_subquery_with_lazyload(self):
        if False:
            print('Hello World!')
        "Mapper load strategy defaults can be upgraded with\n        subqueryload('*') option, while explicit lazyload() option\n        is still honored"
        sess = self._upgrade_fixture()
        users = []
        (User, Order, Item) = self.classes('User', 'Order', 'Item')

        def go():
            if False:
                print('Hello World!')
            users[:] = sess.query(User).options(defaultload(User.orders).defaultload(Order.items).lazyload(Item.keywords)).options(subqueryload('*')).order_by(User.id).all()
        self.assert_sql_count(testing.db, go, 4)

        def go():
            if False:
                print('Hello World!')
            eq_(users, self.static.user_all_result)
        self.assert_sql_count(testing.db, go, 0)

        def go():
            if False:
                return 10
            users[0].orders[0].items[0]
        self.assert_sql_count(testing.db, go, 0)

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[0].orders[0].items[0].keywords
        self.assert_sql_count(testing.db, go, 1)

    def test_subquery_with_joinedload(self):
        if False:
            while True:
                i = 10
        "Mapper load strategy defaults can be upgraded with\n        subqueryload('*') option, while multiple explicit\n        joinedload() options are still honored"
        sess = self._upgrade_fixture()
        users = []

        def go():
            if False:
                for i in range(10):
                    print('nop')
            users[:] = sess.query(self.classes.User).options(joinedload(self.classes.User.addresses)).options(joinedload(self.classes.User.orders)).options(subqueryload('*')).order_by(self.classes.User.id).all()
        self.assert_sql_count(testing.db, go, 3)
        self._assert_fully_loaded(users)

class NoLoadTest(_fixtures.FixtureTest):
    run_inserts = 'once'
    run_deletes = None

    def test_o2m_noload(self):
        if False:
            for i in range(10):
                print('nop')
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        m = self.mapper_registry.map_imperatively(User, users, properties=dict(addresses=relationship(self.mapper_registry.map_imperatively(Address, addresses), lazy='noload')))
        q = fixture_session().query(m)
        result = [None]

        def go():
            if False:
                i = 10
                return i + 15
            x = q.filter(User.id == 7).all()
            x[0].addresses
            result[0] = x
        self.assert_sql_count(testing.db, go, 1)
        self.assert_result(result[0], User, {'id': 7, 'addresses': (Address, [])})

    def test_upgrade_o2m_noload_lazyload_option(self):
        if False:
            for i in range(10):
                print('nop')
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        m = self.mapper_registry.map_imperatively(User, users, properties=dict(addresses=relationship(self.mapper_registry.map_imperatively(Address, addresses), lazy='noload')))
        q = fixture_session().query(m).options(sa.orm.lazyload(User.addresses))
        result = [None]

        def go():
            if False:
                for i in range(10):
                    print('nop')
            x = q.filter(User.id == 7).all()
            x[0].addresses
            result[0] = x
        self.sql_count_(2, go)
        self.assert_result(result[0], User, {'id': 7, 'addresses': (Address, [{'id': 1}])})

    def test_m2o_noload_option(self):
        if False:
            print('Hello World!')
        (Address, addresses, users, User) = (self.classes.Address, self.tables.addresses, self.tables.users, self.classes.User)
        self.mapper_registry.map_imperatively(Address, addresses, properties={'user': relationship(User)})
        self.mapper_registry.map_imperatively(User, users)
        s = fixture_session()
        a1 = s.query(Address).filter_by(id=1).options(sa.orm.noload(Address.user)).first()

        def go():
            if False:
                print('Hello World!')
            eq_(a1.user, None)
        self.sql_count_(0, go)
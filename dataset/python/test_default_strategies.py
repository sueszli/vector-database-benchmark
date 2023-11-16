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
        # verify everything loaded, with no additional sql needed
        def go():
            # comparison with no additional sql
            eq_(users, self.static.user_all_result)
            # keywords are not part of self.static.user_all_result, so
            # verify all the item keywords were loaded, with no more sql.
            # 'any' verifies at least some items have keywords; we build
            # a list for any([...]) instead of any(...) to prove we've
            # iterated all the items with no sql.
            f = util.flatten_iterator
            assert any(
                [
                    i.keywords
                    for i in f([o.items for o in f([u.orders for u in users])])
                ]
            )

        self.assert_sql_count(testing.db, go, 0)

    def _assert_addresses_loaded(self, users):
        # verify all the addresses were joined loaded with no more sql
        def go():
            for u, static in zip(users, self.static.user_all_result):
                eq_(u.addresses, static.addresses)

        self.assert_sql_count(testing.db, go, 0)

    def _downgrade_fixture(self):
        (
            users,
            Keyword,
            items,
            order_items,
            orders,
            Item,
            User,
            Address,
            keywords,
            item_keywords,
            Order,
            addresses,
        ) = (
            self.tables.users,
            self.classes.Keyword,
            self.tables.items,
            self.tables.order_items,
            self.tables.orders,
            self.classes.Item,
            self.classes.User,
            self.classes.Address,
            self.tables.keywords,
            self.tables.item_keywords,
            self.classes.Order,
            self.tables.addresses,
        )

        self.mapper_registry.map_imperatively(Address, addresses)

        self.mapper_registry.map_imperatively(Keyword, keywords)

        self.mapper_registry.map_imperatively(
            Item,
            items,
            properties=dict(
                keywords=relationship(
                    Keyword,
                    secondary=item_keywords,
                    lazy="subquery",
                    order_by=item_keywords.c.keyword_id,
                )
            ),
        )

        self.mapper_registry.map_imperatively(
            Order,
            orders,
            properties=dict(
                items=relationship(
                    Item,
                    secondary=order_items,
                    lazy="subquery",
                    order_by=order_items.c.item_id,
                )
            ),
        )

        self.mapper_registry.map_imperatively(
            User,
            users,
            properties=dict(
                addresses=relationship(
                    Address, lazy="joined", order_by=addresses.c.id
                ),
                orders=relationship(
                    Order, lazy="joined", order_by=orders.c.id
                ),
            ),
        )

        return fixture_session()

    def _upgrade_fixture(self):
        (
            users,
            Keyword,
            items,
            order_items,
            orders,
            Item,
            User,
            Address,
            keywords,
            item_keywords,
            Order,
            addresses,
        ) = (
            self.tables.users,
            self.classes.Keyword,
            self.tables.items,
            self.tables.order_items,
            self.tables.orders,
            self.classes.Item,
            self.classes.User,
            self.classes.Address,
            self.tables.keywords,
            self.tables.item_keywords,
            self.classes.Order,
            self.tables.addresses,
        )

        self.mapper_registry.map_imperatively(Address, addresses)

        self.mapper_registry.map_imperatively(Keyword, keywords)

        self.mapper_registry.map_imperatively(
            Item,
            items,
            properties=dict(
                keywords=relationship(
                    Keyword,
                    secondary=item_keywords,
                    lazy="select",
                    order_by=item_keywords.c.keyword_id,
                )
            ),
        )

        self.mapper_registry.map_imperatively(
            Order,
            orders,
            properties=dict(
                items=relationship(
                    Item,
                    secondary=order_items,
                    lazy=True,
                    order_by=order_items.c.item_id,
                )
            ),
        )

        self.mapper_registry.map_imperatively(
            User,
            users,
            properties=dict(
                addresses=relationship(
                    Address, lazy=True, order_by=addresses.c.id
                ),
                orders=relationship(Order, order_by=orders.c.id),
            ),
        )

        return fixture_session()

    def test_downgrade_baseline(self):
        """Mapper strategy defaults load as expected
        (compare to rest of DefaultStrategyOptionsTest downgrade tests)."""
        sess = self._downgrade_fixture()
        users = []

        # test _downgrade_fixture mapper defaults, 3 queries (2 subquery
        # loads).
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 3)

        # all loaded with no additional sql
        self._assert_fully_loaded(users)

    def test_disable_eagerloads(self):
        """Mapper eager load strategy defaults can be shut off
        with enable_eagerloads(False)."""

        # While this isn't testing a mapper option, it is included
        # as baseline reference for how XYZload('*') option
        # should work, namely, it shouldn't affect later queries
        # (see other test_select_s)
        sess = self._downgrade_fixture()
        users = []

        # demonstrate that enable_eagerloads loads with only 1 sql
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .enable_eagerloads(False)
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # demonstrate that users[0].orders must now be loaded with 3 sql
        # (need to lazyload, and 2 subquery: 3 total)
        def go():
            users[0].orders

        self.assert_sql_count(testing.db, go, 3)

    def test_last_one_wins(self):
        sess = self._downgrade_fixture()
        users = []

        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(subqueryload("*"))
                .options(joinedload(self.classes.User.addresses))
                .options(sa.orm.lazyload("*"))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # verify all the addresses were joined loaded (no more sql)
        self._assert_addresses_loaded(users)

    def test_star_must_be_alone(self):
        self._downgrade_fixture()
        User = self.classes.User

        with expect_raises_message(
            sa.exc.ArgumentError,
            "Wildcard token cannot be followed by another entity",
        ):
            subqueryload("*", User.addresses)

    def test_star_cant_be_followed(self):
        self._downgrade_fixture()
        User = self.classes.User
        Order = self.classes.Order

        with expect_raises_message(
            sa.exc.ArgumentError,
            "Wildcard token cannot be followed by another entity",
        ):
            subqueryload(User.addresses).joinedload("*").selectinload(
                Order.items
            )

    def test_global_star_ignored_no_entities_unbound(self):
        sess = self._downgrade_fixture()
        User = self.classes.User
        opt = sa.orm.lazyload("*")
        q = sess.query(User.name).options(opt)
        eq_(q.all(), [("jack",), ("ed",), ("fred",), ("chuck",)])

    def test_global_star_ignored_no_entities_bound(self):
        sess = self._downgrade_fixture()
        User = self.classes.User
        opt = sa.orm.Load(User).lazyload("*")
        q = sess.query(User.name).options(opt)
        eq_(q.all(), [("jack",), ("ed",), ("fred",), ("chuck",)])

    def test_select_with_joinedload(self):
        """Mapper load strategy defaults can be downgraded with
        lazyload('*') option, while explicit joinedload() option
        is still honored"""
        sess = self._downgrade_fixture()
        users = []

        # lazyload('*') shuts off 'orders' subquery: only 1 sql
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(sa.orm.lazyload("*"))
                .options(joinedload(self.classes.User.addresses))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # verify all the addresses were joined loaded (no more sql)
        self._assert_addresses_loaded(users)

        # users[0] has orders, which need to lazy load, and 2 subquery:
        # (same as with test_disable_eagerloads): 3 total sql
        def go():
            users[0].orders

        self.assert_sql_count(testing.db, go, 3)

    def test_select_with_subqueryload(self):
        """Mapper load strategy defaults can be downgraded with
        lazyload('*') option, while explicit subqueryload() option
        is still honored"""
        sess = self._downgrade_fixture()
        users = []

        # now test 'default_strategy' option combined with 'subquery'
        # shuts off 'addresses' load AND orders.items load: 2 sql expected
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(sa.orm.lazyload("*"))
                .options(subqueryload(self.classes.User.orders))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 2)

        # Verify orders have already been loaded: 0 sql
        def go():
            for u, static in zip(users, self.static.user_all_result):
                assert len(u.orders) == len(static.orders)

        self.assert_sql_count(testing.db, go, 0)

        # Verify lazyload('*') prevented orders.items load
        # users[0].orders[0] has 3 items, each with keywords: 2 sql
        # ('items' and 'items.keywords' subquery)
        # but!  the subqueryload for further sub-items *does* load.
        # so at the moment the wildcard load is shut off for this load
        def go():
            for i in users[0].orders[0].items:
                i.keywords

        self.assert_sql_count(testing.db, go, 2)

        # lastly, make sure they actually loaded properly
        eq_(users, self.static.user_all_result)

    def test_noload_with_joinedload(self):
        """Mapper load strategy defaults can be downgraded with
        noload('*') option, while explicit joinedload() option
        is still honored"""
        sess = self._downgrade_fixture()
        users = []

        # test noload('*') shuts off 'orders' subquery, only 1 sql
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(sa.orm.noload("*"))
                .options(joinedload(self.classes.User.addresses))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # verify all the addresses were joined loaded (no more sql)
        self._assert_addresses_loaded(users)

        # User.orders should have loaded "noload" (meaning [])
        def go():
            for u in users:
                assert u.orders == []

        self.assert_sql_count(testing.db, go, 0)

    def test_noload_with_subqueryload(self):
        """Mapper load strategy defaults can be downgraded with
        noload('*') option, while explicit subqueryload() option
        is still honored"""
        sess = self._downgrade_fixture()
        users = []

        # test noload('*') option combined with subqueryload()
        # shuts off 'addresses' load AND orders.items load: 2 sql expected
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(sa.orm.noload("*"))
                .options(subqueryload(self.classes.User.orders))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 2)

        def go():
            # Verify orders have already been loaded: 0 sql
            for u, static in zip(users, self.static.user_all_result):
                assert len(u.orders) == len(static.orders)
            # Verify noload('*') prevented orders.items load
            # and set 'items' to []
            for u in users:
                for o in u.orders:
                    assert o.items == []

        self.assert_sql_count(testing.db, go, 0)

    def test_joined(self):
        """Mapper load strategy defaults can be upgraded with
        joinedload('*') option."""
        sess = self._upgrade_fixture()
        users = []

        # test upgrade all to joined: 1 sql
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(joinedload("*"))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # verify everything loaded, with no additional sql needed
        self._assert_fully_loaded(users)

    def test_joined_path_wildcards(self):
        sess = self._upgrade_fixture()
        users = []

        User, Order, Item = self.classes("User", "Order", "Item")

        # test upgrade all to joined: 1 sql
        def go():
            users[:] = (
                sess.query(User)
                .options(joinedload("*"))
                .options(defaultload(User.addresses).joinedload("*"))
                .options(defaultload(User.orders).joinedload("*"))
                .options(
                    defaultload(User.orders)
                    .defaultload(Order.items)
                    .joinedload("*")
                )
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)
        self._assert_fully_loaded(users)

    def test_joined_with_lazyload(self):
        """Mapper load strategy defaults can be upgraded with
        joinedload('*') option, while explicit lazyload() option
        is still honored"""
        sess = self._upgrade_fixture()
        users = []

        User, Order, Item = self.classes("User", "Order", "Item")

        # test joined all but 'keywords': upgraded to 1 sql
        def go():
            users[:] = (
                sess.query(User)
                .options(
                    defaultload(User.orders)
                    .defaultload(Order.items)
                    .lazyload(Item.keywords)
                )
                .options(joinedload("*"))
                .order_by(User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 1)

        # everything (but keywords) loaded ok
        # (note self.static.user_all_result contains no keywords)
        def go():
            eq_(users, self.static.user_all_result)

        self.assert_sql_count(testing.db, go, 0)

        # verify the items were loaded, while item.keywords were not
        def go():
            # redundant with last test, but illustrative
            users[0].orders[0].items[0]

        self.assert_sql_count(testing.db, go, 0)

        def go():
            users[0].orders[0].items[0].keywords

        self.assert_sql_count(testing.db, go, 1)

    def test_joined_with_subqueryload(self):
        """Mapper load strategy defaults can be upgraded with
        joinedload('*') option, while explicit subqueryload() option
        is still honored"""
        sess = self._upgrade_fixture()
        users = []

        # test upgrade all but 'addresses', which is subquery loaded (2 sql)
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(subqueryload(self.classes.User.addresses))
                .options(joinedload("*"))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 2)

        # verify everything loaded, with no additional sql needed
        self._assert_fully_loaded(users)

    def test_subquery(self):
        """Mapper load strategy defaults can be upgraded with
        subqueryload('*') option."""
        sess = self._upgrade_fixture()
        users = []

        # test upgrade all to subquery: 1 sql + 4 relationships = 5
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(subqueryload("*"))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 5)

        # verify everything loaded, with no additional sql needed
        self._assert_fully_loaded(users)

    def test_subquery_path_wildcards(self):
        sess = self._upgrade_fixture()
        users = []

        User, Order = self.classes("User", "Order")

        # test upgrade all to subquery: 1 sql + 4 relationships = 5
        def go():
            users[:] = (
                sess.query(User)
                .options(subqueryload("*"))
                .options(defaultload(User.addresses).subqueryload("*"))
                .options(defaultload(User.orders).subqueryload("*"))
                .options(
                    defaultload(User.orders)
                    .defaultload(Order.items)
                    .subqueryload("*")
                )
                .order_by(User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 5)

        # verify everything loaded, with no additional sql needed
        self._assert_fully_loaded(users)

    def test_subquery_with_lazyload(self):
        """Mapper load strategy defaults can be upgraded with
        subqueryload('*') option, while explicit lazyload() option
        is still honored"""
        sess = self._upgrade_fixture()
        users = []
        User, Order, Item = self.classes("User", "Order", "Item")

        # test subquery all but 'keywords' (1 sql + 3 relationships = 4)
        def go():
            users[:] = (
                sess.query(User)
                .options(
                    defaultload(User.orders)
                    .defaultload(Order.items)
                    .lazyload(Item.keywords)
                )
                .options(subqueryload("*"))
                .order_by(User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 4)

        # no more sql
        # (note self.static.user_all_result contains no keywords)
        def go():
            eq_(users, self.static.user_all_result)

        self.assert_sql_count(testing.db, go, 0)

        # verify the item.keywords were not loaded
        def go():
            users[0].orders[0].items[0]

        self.assert_sql_count(testing.db, go, 0)

        def go():
            users[0].orders[0].items[0].keywords

        self.assert_sql_count(testing.db, go, 1)

    def test_subquery_with_joinedload(self):
        """Mapper load strategy defaults can be upgraded with
        subqueryload('*') option, while multiple explicit
        joinedload() options are still honored"""
        sess = self._upgrade_fixture()
        users = []

        # test upgrade all but 'addresses' & 'orders', which are joinedloaded
        # (1 sql + items + keywords = 3)
        def go():
            users[:] = (
                sess.query(self.classes.User)
                .options(joinedload(self.classes.User.addresses))
                .options(joinedload(self.classes.User.orders))
                .options(subqueryload("*"))
                .order_by(self.classes.User.id)
                .all()
            )

        self.assert_sql_count(testing.db, go, 3)

        # verify everything loaded, with no additional sql needed
        self._assert_fully_loaded(users)


class NoLoadTest(_fixtures.FixtureTest):
    run_inserts = "once"
    run_deletes = None

    def test_o2m_noload(self):
        Address, addresses, users, User = (
            self.classes.Address,
            self.tables.addresses,
            self.tables.users,
            self.classes.User,
        )

        m = self.mapper_registry.map_imperatively(
            User,
            users,
            properties=dict(
                addresses=relationship(
                    self.mapper_registry.map_imperatively(Address, addresses),
                    lazy="noload",
                )
            ),
        )
        q = fixture_session().query(m)
        result = [None]

        def go():
            x = q.filter(User.id == 7).all()
            x[0].addresses
            result[0] = x

        self.assert_sql_count(testing.db, go, 1)

        self.assert_result(
            result[0], User, {"id": 7, "addresses": (Address, [])}
        )

    def test_upgrade_o2m_noload_lazyload_option(self):
        Address, addresses, users, User = (
            self.classes.Address,
            self.tables.addresses,
            self.tables.users,
            self.classes.User,
        )

        m = self.mapper_registry.map_imperatively(
            User,
            users,
            properties=dict(
                addresses=relationship(
                    self.mapper_registry.map_imperatively(Address, addresses),
                    lazy="noload",
                )
            ),
        )
        q = fixture_session().query(m).options(sa.orm.lazyload(User.addresses))
        result = [None]

        def go():
            x = q.filter(User.id == 7).all()
            x[0].addresses
            result[0] = x

        self.sql_count_(2, go)

        self.assert_result(
            result[0], User, {"id": 7, "addresses": (Address, [{"id": 1}])}
        )

    def test_m2o_noload_option(self):
        Address, addresses, users, User = (
            self.classes.Address,
            self.tables.addresses,
            self.tables.users,
            self.classes.User,
        )
        self.mapper_registry.map_imperatively(
            Address, addresses, properties={"user": relationship(User)}
        )
        self.mapper_registry.map_imperatively(User, users)
        s = fixture_session()
        a1 = (
            s.query(Address)
            .filter_by(id=1)
            .options(sa.orm.noload(Address.user))
            .first()
        )

        def go():
            eq_(a1.user, None)

        self.sql_count_(0, go)

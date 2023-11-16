from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.orm import Session
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.entities import ComparableEntity
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table


class InheritingSelectablesTest(fixtures.MappedTest):
    @classmethod
    def define_tables(cls, metadata):
        foo = Table(
            "foo",
            metadata,
            Column("a", String(30), primary_key=1),
            Column("b", String(30), nullable=0),
        )

        cls.tables.bar = foo.select().where(foo.c.b == "bar").alias("bar")
        cls.tables.baz = foo.select().where(foo.c.b == "baz").alias("baz")

    def test_load(self, connection):
        foo, bar, baz = self.tables.foo, self.tables.bar, self.tables.baz
        # TODO: add persistence test also
        connection.execute(foo.insert(), dict(a="not bar", b="baz"))
        connection.execute(foo.insert(), dict(a="also not bar", b="baz"))
        connection.execute(foo.insert(), dict(a="i am bar", b="bar"))
        connection.execute(foo.insert(), dict(a="also bar", b="bar"))

        class Foo(ComparableEntity):
            pass

        class Bar(Foo):
            pass

        class Baz(Foo):
            pass

        self.mapper_registry.map_imperatively(Foo, foo, polymorphic_on=foo.c.b)

        self.mapper_registry.map_imperatively(
            Baz,
            baz,
            with_polymorphic=(
                "*",
                foo.join(baz, foo.c.b == "baz").select().subquery("baz"),
            ),
            inherits=Foo,
            inherit_condition=(foo.c.a == baz.c.a),
            inherit_foreign_keys=[baz.c.a],
            polymorphic_identity="baz",
        )

        self.mapper_registry.map_imperatively(
            Bar,
            bar,
            with_polymorphic=(
                "*",
                foo.join(bar, foo.c.b == "bar").select().subquery("bar"),
            ),
            inherits=Foo,
            inherit_condition=(foo.c.a == bar.c.a),
            inherit_foreign_keys=[bar.c.a],
            polymorphic_identity="bar",
        )

        s = Session(connection)
        eq_(s.query(Bar).all(), [Bar(), Bar()])


class JoinFromSelectPersistenceTest(fixtures.MappedTest):
    """test for [ticket:2885]"""

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "base",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("type", String(50)),
        )
        Table(
            "child",
            metadata,
            # 1. name of column must be different, so that we rely on
            # mapper._table_to_equated to link the two cols
            Column(
                "child_id", Integer, ForeignKey("base.id"), primary_key=True
            ),
            Column("name", String(50)),
        )

    @classmethod
    def setup_classes(cls):
        class Base(cls.Comparable):
            pass

        class Child(Base):
            pass

    def test_map_to_select(self):
        Base, Child = self.classes.Base, self.classes.Child
        base, child = self.tables.base, self.tables.child

        base_select = select(base).alias()
        self.mapper_registry.map_imperatively(
            Base,
            base_select,
            polymorphic_on=base_select.c.type,
            polymorphic_identity="base",
        )
        self.mapper_registry.map_imperatively(
            Child, child, inherits=Base, polymorphic_identity="child"
        )

        sess = fixture_session()

        # 2. use an id other than "1" here so can't rely on
        # the two inserts having the same id
        c1 = Child(id=12, name="c1")
        sess.add(c1)

        sess.commit()
        sess.close()

        c1 = sess.query(Child).one()
        eq_(c1.name, "c1")

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import backref
from sqlalchemy.orm import configure_mappers
from sqlalchemy.orm import relationship
from sqlalchemy.testing import fixtures
from sqlalchemy.testing.fixtures import fixture_session
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table


class PolymorphicCircularTest(fixtures.MappedTest):
    run_setup_mappers = "once"

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "table1",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column(
                "related_id", Integer, ForeignKey("table1.id"), nullable=True
            ),
            Column("type", String(30)),
            Column("name", String(30)),
        )

        Table(
            "table2",
            metadata,
            Column("id", Integer, ForeignKey("table1.id"), primary_key=True),
        )

        Table(
            "table3",
            metadata,
            Column("id", Integer, ForeignKey("table1.id"), primary_key=True),
        )

        Table(
            "data",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("node_id", Integer, ForeignKey("table1.id")),
            Column("data", String(30)),
        )

    @classmethod
    def setup_mappers(cls):
        table1, table2, table3, data = cls.tables(
            "table1", "table2", "table3", "data"
        )

        Base = cls.Basic

        class Table1(Base):
            def __init__(self, name, data=None):
                self.name = name
                if data is not None:
                    self.data = data

            def __repr__(self):
                return "%s(%s, %s, %s)" % (
                    self.__class__.__name__,
                    self.id,
                    repr(str(self.name)),
                    repr(self.data),
                )

        class Table1B(Table1):
            pass

        class Table2(Table1):
            pass

        class Table3(Table1):
            pass

        class Data(Base):
            def __init__(self, data):
                self.data = data

            def __repr__(self):
                return "%s(%s, %s)" % (
                    self.__class__.__name__,
                    self.id,
                    repr(str(self.data)),
                )

        # currently, the "eager" relationships degrade to lazy relationships
        # due to the polymorphic load.
        # the "nxt" relationship used to have a "lazy='joined'" on it, but the
        # EagerLoader raises the "self-referential"
        # exception now.  since eager loading would never work for that
        # relationship anyway, its better that the user
        # gets an exception instead of it silently not eager loading.
        # NOTE: using "nxt" instead of "next" to avoid 2to3 turning it into
        # __next__() for some reason.
        table1_mapper = cls.mapper_registry.map_imperatively(
            Table1,
            table1,
            # select_table=join,
            polymorphic_on=table1.c.type,
            polymorphic_identity="table1",
            properties={
                "nxt": relationship(
                    Table1,
                    backref=backref(
                        "prev", remote_side=table1.c.id, uselist=False
                    ),
                    uselist=False,
                    primaryjoin=table1.c.id == table1.c.related_id,
                ),
                "data": relationship(
                    cls.mapper_registry.map_imperatively(Data, data),
                    lazy="joined",
                    order_by=data.c.id,
                ),
            },
        )

        cls.mapper_registry.map_imperatively(
            Table1B, inherits=table1_mapper, polymorphic_identity="table1b"
        )

        cls.mapper_registry.map_imperatively(
            Table2,
            table2,
            inherits=table1_mapper,
            polymorphic_identity="table2",
        )

        cls.mapper_registry.map_imperatively(
            Table3,
            table3,
            inherits=table1_mapper,
            polymorphic_identity="table3",
        )

        configure_mappers()
        assert table1_mapper.primary_key == (
            table1.c.id,
        ), table1_mapper.primary_key

    def test_one(self):
        Table1, Table2 = self.classes("Table1", "Table2")
        self._testlist([Table1, Table2, Table1, Table2])

    def test_two(self):
        Table3 = self.classes.Table3
        self._testlist([Table3])

    def test_three(self):
        Table1, Table1B, Table2, Table3 = self.classes(
            "Table1", "Table1B", "Table2", "Table3"
        )
        self._testlist(
            [
                Table2,
                Table1,
                Table1B,
                Table3,
                Table3,
                Table1B,
                Table1B,
                Table2,
                Table1,
            ]
        )

    def test_four(self):
        Table1, Table1B, Table2, Table3, Data = self.classes(
            "Table1", "Table1B", "Table2", "Table3", "Data"
        )
        self._testlist(
            [
                Table2("t2", [Data("data1"), Data("data2")]),
                Table1("t1", []),
                Table3("t3", [Data("data3")]),
                Table1B("t1b", [Data("data4"), Data("data5")]),
            ]
        )

    def _testlist(self, classes):
        Table1 = self.classes.Table1

        sess = fixture_session()

        # create objects in a linked list
        count = 1
        obj = None
        for c in classes:
            if isinstance(c, type):
                newobj = c("item %d" % count)
                count += 1
            else:
                newobj = c
            if obj is not None:
                obj.nxt = newobj
            else:
                t = newobj
            obj = newobj

        # save to DB
        sess.add(t)
        sess.flush()

        # string version of the saved list
        assertlist = []
        node = t
        while node:
            assertlist.append(node)
            n = node.nxt
            if n is not None:
                assert n.prev is node
            node = n
        original = repr(assertlist)

        # clear and query forwards
        sess.expunge_all()
        node = (
            sess.query(Table1)
            .order_by(Table1.id)
            .filter(Table1.id == t.id)
            .first()
        )
        assertlist = []
        while node:
            assertlist.append(node)
            n = node.nxt
            if n is not None:
                assert n.prev is node
            node = n
        forwards = repr(assertlist)

        # clear and query backwards
        sess.expunge_all()
        node = (
            sess.query(Table1)
            .order_by(Table1.id)
            .filter(Table1.id == obj.id)
            .first()
        )
        assertlist = []
        while node:
            assertlist.insert(0, node)
            n = node.prev
            if n is not None:
                assert n.nxt is node
            node = n
        backwards = repr(assertlist)

        # everything should match !
        assert original == forwards == backwards

from sqlalchemy import Column
from sqlalchemy import column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import join
from sqlalchemy import lateral
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy import true
from sqlalchemy.engine import default
from sqlalchemy.sql import func
from sqlalchemy.sql import select
from sqlalchemy.sql.selectable import Lateral
from sqlalchemy.testing import assert_raises_message
from sqlalchemy.testing import AssertsCompiledSQL
from sqlalchemy.testing import fixtures

class LateralTest(fixtures.TablesTest, AssertsCompiledSQL):
    __dialect__ = default.DefaultDialect(supports_native_boolean=True)
    assert_from_linting = True
    run_setup_bind = None
    run_create_tables = None

    @classmethod
    def define_tables(cls, metadata):
        if False:
            print('Hello World!')
        Table('people', metadata, Column('people_id', Integer, primary_key=True), Column('age', Integer), Column('name', String(30)))
        Table('bookcases', metadata, Column('bookcase_id', Integer, primary_key=True), Column('bookcase_owner_id', Integer, ForeignKey('people.people_id')), Column('bookcase_shelves', Integer), Column('bookcase_width', Integer))
        Table('books', metadata, Column('book_id', Integer, primary_key=True), Column('bookcase_id', Integer, ForeignKey('bookcases.bookcase_id')), Column('book_owner_id', Integer, ForeignKey('people.people_id')), Column('book_weight', Integer))

    def test_standalone(self):
        if False:
            for i in range(10):
                print('nop')
        table1 = self.tables.people
        subq = select(table1.c.people_id).subquery()
        self.assert_compile(lateral(subq, name='alias'), 'LATERAL (SELECT people.people_id FROM people)')
        self.assert_compile(subq.lateral(name='alias'), 'LATERAL (SELECT people.people_id FROM people)')

    def test_standalone_implicit_subquery(self):
        if False:
            while True:
                i = 10
        table1 = self.tables.people
        subq = select(table1.c.people_id)
        self.assert_compile(lateral(subq, name='alias'), 'LATERAL (SELECT people.people_id FROM people)')
        self.assert_compile(subq.lateral(name='alias'), 'LATERAL (SELECT people.people_id FROM people)')

    def test_select_from(self):
        if False:
            print('Hello World!')
        table1 = self.tables.people
        subq = select(table1.c.people_id).subquery()
        self.assert_compile(select(subq.lateral(name='alias')), 'SELECT alias.people_id FROM LATERAL (SELECT people.people_id AS people_id FROM people) AS alias')

    def test_alias_of_lateral(self):
        if False:
            for i in range(10):
                print('nop')
        table1 = self.tables.people
        subq = select(table1.c.people_id).subquery()
        self.assert_compile(select(subq.lateral().alias(name='alias')), 'SELECT alias.people_id FROM LATERAL (SELECT people.people_id AS people_id FROM people) AS alias')

    def test_select_from_implicit_subquery(self):
        if False:
            return 10
        table1 = self.tables.people
        subq = select(table1.c.people_id)
        self.assert_compile(select(subq.lateral(name='alias')), 'SELECT alias.people_id FROM LATERAL (SELECT people.people_id AS people_id FROM people) AS alias')

    def test_select_from_text_implicit_subquery(self):
        if False:
            print('Hello World!')
        table1 = self.tables.people
        subq = text('SELECT people_id FROM people').columns(table1.c.people_id)
        self.assert_compile(select(subq.lateral(name='alias')), 'SELECT alias.people_id FROM LATERAL (SELECT people_id FROM people) AS alias')

    def test_plain_join(self):
        if False:
            print('Hello World!')
        table1 = self.tables.people
        table2 = self.tables.books
        subq = select(table2.c.book_id).where(table2.c.book_owner_id == table1.c.people_id)
        self.assert_compile(join(table1, lateral(subq.subquery(), name='alias'), true()), 'people JOIN LATERAL (SELECT books.book_id AS book_id FROM books, people WHERE books.book_owner_id = people.people_id) AS alias ON true')
        self.assert_compile(select(table1).select_from(join(table1, lateral(subq.subquery(), name='alias'), true())), 'SELECT people.people_id, people.age, people.name FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE books.book_owner_id = people.people_id) AS alias ON true')
        subq = subq.correlate(table1)
        self.assert_compile(select(table1).select_from(join(table1, lateral(subq.subquery(), name='alias'), true())), 'SELECT people.people_id, people.age, people.name FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE books.book_owner_id = people.people_id) AS alias ON true')

    def test_plain_join_implicit_subquery(self):
        if False:
            for i in range(10):
                print('nop')
        table1 = self.tables.people
        table2 = self.tables.books
        subq = select(table2.c.book_id).where(table2.c.book_owner_id == table1.c.people_id)
        self.assert_compile(join(table1, lateral(subq, name='alias'), true()), 'people JOIN LATERAL (SELECT books.book_id AS book_id FROM books, people WHERE books.book_owner_id = people.people_id) AS alias ON true')
        self.assert_compile(select(table1).select_from(join(table1, lateral(subq, name='alias'), true())), 'SELECT people.people_id, people.age, people.name FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE books.book_owner_id = people.people_id) AS alias ON true')
        subq = subq.correlate(table1)
        self.assert_compile(select(table1).select_from(join(table1, lateral(subq, name='alias'), true())), 'SELECT people.people_id, people.age, people.name FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE books.book_owner_id = people.people_id) AS alias ON true')

    def test_join_lateral_w_select_subquery(self):
        if False:
            for i in range(10):
                print('nop')
        table1 = self.tables.people
        table2 = self.tables.books
        subq = select(table2.c.book_id).correlate(table1).where(table1.c.people_id == table2.c.book_owner_id).subquery().lateral()
        stmt = select(table1, subq.c.book_id).select_from(table1.join(subq, true()))
        self.assert_compile(stmt, 'SELECT people.people_id, people.age, people.name, anon_1.book_id FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE people.people_id = books.book_owner_id) AS anon_1 ON true')

    @testing.combinations((True,), (False,))
    def test_join_lateral_subquery_twolevel(self, use_twolevel):
        if False:
            i = 10
            return i + 15
        (people, books, bookcases) = self.tables('people', 'books', 'bookcases')
        p1 = select(books.c.book_id, books.c.bookcase_id, books.c.book_owner_id).subquery()
        p2 = select(bookcases.c.bookcase_id, bookcases.c.bookcase_owner_id).where(bookcases.c.bookcase_id == p1.c.bookcase_id).subquery().lateral()
        p3 = select(people.c.people_id).where(p1.c.book_owner_id == people.c.people_id).subquery().lateral()
        onelevel = select(p1.c.book_id, p2.c.bookcase_id).select_from(p1).join(p2, true())
        if use_twolevel:
            twolevel = onelevel.add_columns(p3.c.people_id).join(p3, true())
            self.assert_compile(twolevel, 'SELECT anon_1.book_id, anon_2.bookcase_id, anon_3.people_id FROM (SELECT books.book_id AS book_id, books.bookcase_id AS bookcase_id, books.book_owner_id AS book_owner_id FROM books) AS anon_1 JOIN LATERAL (SELECT bookcases.bookcase_id AS bookcase_id, bookcases.bookcase_owner_id AS bookcase_owner_id FROM bookcases WHERE bookcases.bookcase_id = anon_1.bookcase_id) AS anon_2 ON true JOIN LATERAL (SELECT people.people_id AS people_id FROM people WHERE anon_1.book_owner_id = people.people_id) AS anon_3 ON true')
        else:
            self.assert_compile(onelevel, 'SELECT anon_1.book_id, anon_2.bookcase_id FROM (SELECT books.book_id AS book_id, books.bookcase_id AS bookcase_id, books.book_owner_id AS book_owner_id FROM books) AS anon_1 JOIN LATERAL (SELECT bookcases.bookcase_id AS bookcase_id, bookcases.bookcase_owner_id AS bookcase_owner_id FROM bookcases WHERE bookcases.bookcase_id = anon_1.bookcase_id) AS anon_2 ON true')

    def test_join_lateral_w_select_implicit_subquery(self):
        if False:
            i = 10
            return i + 15
        table1 = self.tables.people
        table2 = self.tables.books
        subq = select(table2.c.book_id).correlate(table1).where(table1.c.people_id == table2.c.book_owner_id).lateral()
        stmt = select(table1, subq.c.book_id).select_from(table1.join(subq, true()))
        self.assert_compile(stmt, 'SELECT people.people_id, people.age, people.name, anon_1.book_id FROM people JOIN LATERAL (SELECT books.book_id AS book_id FROM books WHERE people.people_id = books.book_owner_id) AS anon_1 ON true')

    def test_from_function(self):
        if False:
            i = 10
            return i + 15
        bookcases = self.tables.bookcases
        srf = lateral(func.generate_series(1, bookcases.c.bookcase_shelves))
        self.assert_compile(select(bookcases).select_from(bookcases.join(srf, true())), 'SELECT bookcases.bookcase_id, bookcases.bookcase_owner_id, bookcases.bookcase_shelves, bookcases.bookcase_width FROM bookcases JOIN LATERAL generate_series(:generate_series_1, bookcases.bookcase_shelves) AS anon_1 ON true')

    def test_no_alias_construct(self):
        if False:
            for i in range(10):
                print('nop')
        a = table('a', column('x'))
        assert_raises_message(NotImplementedError, 'The Lateral class is not intended to be constructed directly.  Please use the lateral\\(\\) standalone', Lateral, a, 'foo')
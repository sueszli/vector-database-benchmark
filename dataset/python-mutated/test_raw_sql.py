from __future__ import absolute_import, print_function, division
import unittest
from datetime import date
from pony.orm import *
from pony.orm.tests.testutils import raises_exception
from pony.orm.tests import setup_database, teardown_database, only_for
db = Database()

class Person(db.Entity):
    id = PrimaryKey(int)
    name = Required(str)
    age = Required(int)
    dob = Required(date)

@only_for('sqlite')
class TestRawSQL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        setup_database(db)
        with db_session:
            Person(id=1, name='John', age=30, dob=date(1985, 1, 1))
            Person(id=2, name='Mike', age=32, dob=date(1983, 5, 20))
            Person(id=3, name='Mary', age=20, dob=date(1995, 2, 15))

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        teardown_database(db)

    @db_session
    def test_1(self):
        if False:
            i = 10
            return i + 15
        persons = select((p for p in Person if raw_sql('abs("p"."age") > 25')))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_2(self):
        if False:
            while True:
                i = 10
        persons = select((p for p in Person if raw_sql('abs("p"."age")') > 25))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_3(self):
        if False:
            return 10
        x = 25
        persons = select((p for p in Person if raw_sql('abs("p"."age") > $x')))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_4(self):
        if False:
            print('Hello World!')
        x = 1
        s = 'p.id > $x'
        persons = select((p for p in Person if raw_sql(s)))[:]
        self.assertEqual(set(persons), {Person[2], Person[3]})

    @db_session
    def test_5(self):
        if False:
            while True:
                i = 10
        x = 1
        cond = raw_sql('p.id > $x')
        persons = select((p for p in Person if cond))[:]
        self.assertEqual(set(persons), {Person[2], Person[3]})

    @db_session
    def test_6(self):
        if False:
            print('Hello World!')
        x = date(1990, 1, 1)
        persons = select((p for p in Person if raw_sql('p.dob < $x')))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_7(self):
        if False:
            while True:
                i = 10
        x = 10
        y = 15
        persons = select((p for p in Person if raw_sql('p.age > $(x + y)')))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_8(self):
        if False:
            while True:
                i = 10
        persons = select((p for p in Person if raw_sql('p.dob < $date.today()')))[:]
        self.assertEqual(set(persons), {Person[1], Person[2], Person[3]})

    @db_session
    def test_9(self):
        if False:
            for i in range(10):
                print('nop')
        names = select((raw_sql('UPPER(p.name)') for p in Person))[:]
        self.assertEqual(set(names), {'JOHN', 'MIKE', 'MARY'})

    @db_session
    def test_10(self):
        if False:
            i = 10
            return i + 15
        dates = select((raw_sql('(p.dob)') for p in Person)).order_by(lambda : p.id)[:]
        self.assertEqual(dates, ['1985-01-01', '1983-05-20', '1995-02-15'])

    @db_session
    def test_11(self):
        if False:
            print('Hello World!')
        dates = select((raw_sql('(p.dob)', result_type=date) for p in Person)).order_by(lambda : p.id)[:]
        self.assertEqual(dates, [date(1985, 1, 1), date(1983, 5, 20), date(1995, 2, 15)])

    @db_session
    def test_12(self):
        if False:
            print('Hello World!')
        x = 25
        persons = Person.select(lambda p: p.age > raw_sql('$x'))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_13(self):
        if False:
            for i in range(10):
                print('nop')
        x = 25
        persons = select((p for p in Person)).filter(lambda p: p.age > raw_sql('$x'))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_14(self):
        if False:
            i = 10
            return i + 15
        x = 25
        persons = Person.select().filter(raw_sql('p.age > $x'))[:]
        self.assertEqual(set(persons), {Person[1], Person[2]})

    @db_session
    def test_15(self):
        if False:
            i = 10
            return i + 15
        x = '123'
        y = 'John'
        persons = Person.select(lambda p: raw_sql('UPPER(p.name) || $x') == raw_sql("UPPER($y || '123')"))[:]
        self.assertEqual(set(persons), {Person[1]})

    @db_session
    def test_16(self):
        if False:
            for i in range(10):
                print('nop')
        x = 10
        y = 31
        q = select((p for p in Person if p.age > x and p.age < raw_sql('$y')))
        x = date(1980, 1, 1)
        y = 'j'
        q = q.filter(lambda p: p.dob > x and p.name.startswith(raw_sql('UPPER($y)')))
        persons = q[:]
        self.assertEqual(set(persons), {Person[1]})

    @db_session
    def test_17(self):
        if False:
            return 10
        x = 9
        persons = Person.select().order_by(lambda p: raw_sql('SUBSTR(p.dob, $x)'))[:]
        self.assertEqual(persons, [Person[1], Person[3], Person[2]])

    @db_session
    def test_18(self):
        if False:
            return 10
        x = 9
        persons = Person.select().order_by(raw_sql('SUBSTR(p.dob, $x)'))[:]
        self.assertEqual(persons, [Person[1], Person[3], Person[2]])

    @db_session
    @raises_exception(TranslationError, 'Expression `raw_sql(p.name)` cannot be translated into SQL because raw SQL fragment will be different for each row')
    def test_19(self):
        if False:
            i = 10
            return i + 15
        select((p for p in Person if raw_sql(p.name)))[:]

    @db_session
    @raises_exception(ExprEvalError, "`raw_sql('p.dob < $x')` raises NameError: name 'x' is not defined")
    def test_20(self):
        if False:
            return 10
        select((p for p in Person if raw_sql('p.dob < $x')))[:]

    @db_session
    def test_21(self):
        if False:
            for i in range(10):
                print('nop')
        x = None
        persons = select((p for p in Person if p.id == 1 and raw_sql('$x') is None))[:]
        self.assertEqual(persons, [Person[1]])
if __name__ == '__main__':
    unittest.main()
import unittest
from pony.orm import *
from pony.orm.tests.testutils import *
from pony.orm.tests import setup_database, teardown_database
db = Database()

class Country(db.Entity):
    id = PrimaryKey(int)
    name = Required(str)
    persons = Set('Person')

class Person(db.Entity):
    id = PrimaryKey(int)
    name = Required(str)
    country = Required(Country)

class TestProxy(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        setup_database(db)
        with db_session:
            c1 = Country(id=1, name='Russia')
            c2 = Country(id=2, name='Japan')
            Person(id=1, name='Alexander Nevskiy', country=c1)
            Person(id=2, name='Raikou Minamoto', country=c2)
            Person(id=3, name='Ibaraki Douji', country=c2)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        teardown_database(db)

    def test_1(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            p = make_proxy(Person[2])
        with db_session:
            x1 = db.local_stats[None].db_count
            name = p.name
            country = p.country
            x2 = db.local_stats[None].db_count
        self.assertEqual(x1, x2 - 1)

    def test_2(self):
        if False:
            print('Hello World!')
        with db_session:
            p = make_proxy(Person[2])
            name = p.name
            country = p.country
        with db_session:
            x1 = db.local_stats[None].db_count
            name = p.name
            country = p.country
            x2 = db.local_stats[None].db_count
        self.assertEqual(x1, x2 - 1)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            p = Person[2]
            proxy = make_proxy(p)
        with db_session:
            p2 = Person[2]
            name1 = 'Tamamo no Mae'
            p2.name = name1
            name2 = proxy.name
        self.assertEqual(name1, name2)

    def test_4(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            p = Person[2]
            proxy = make_proxy(p)
        with db_session:
            p2 = Person[2]
            name1 = 'Tamamo no Mae'
            p2.name = name1
        with db_session:
            name2 = proxy.name
        self.assertEqual(name1, name2)

    def test_5(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            p = Person[2]
            r = repr(p)
            self.assertEqual(r, 'Person[2]')
            proxy = make_proxy(p)
            r = repr(proxy)
            self.assertEqual(r, '<EntityProxy(Person[2])>')
        r = repr(proxy)
        self.assertEqual(r, '<EntityProxy(Person[2])>')
        del p
        r = repr(proxy)
        self.assertEqual(r, '<EntityProxy(Person[2])>')

    def test_6(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            p = Person[2]
            proxy = make_proxy(p)
            proxy.name = 'Okita Souji'
            self.assertEqual(proxy.name, 'Okita Souji')
            self.assertEqual(p.name, 'Okita Souji')

    def test_7(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            p = Person[2]
            proxy = make_proxy(p)
            proxy.name = 'Okita Souji'
            self.assertEqual(proxy.name, 'Okita Souji')
            self.assertEqual(p.name, 'Okita Souji')

    def test_8(self):
        if False:
            return 10
        with db_session:
            c1 = Country[1]
            c1_proxy = make_proxy(c1)
            p2 = Person[2]
            self.assertNotEqual(p2.country, c1)
            self.assertNotEqual(p2.country, c1_proxy)
            p2.country = c1_proxy
            self.assertEqual(p2.country, c1_proxy)
            self.assertIs(p2.country, c1)

    def test_9(self):
        if False:
            while True:
                i = 10
        with db_session:
            c2 = Country[2]
            c2_proxy = make_proxy(c2)
            persons = select((p for p in Person if p.country == c2_proxy))
            self.assertEqual({p.id for p in persons}, {2, 3})
if __name__ == '__main__':
    unittest.main()
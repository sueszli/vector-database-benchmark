from __future__ import absolute_import, print_function, division
from binascii import unhexlify
import unittest
from pony.orm.core import *
from pony.orm.tests.testutils import *
from pony.orm.tests import only_for
db = Database('sqlite', ':memory:')

class Person(db.Entity):
    name = Required(str)
    age = Optional(int)
    image = Optional(buffer)
db.generate_mapping(create_tables=True)
with db_session:
    p1 = Person(name='John', age=20, image=unhexlify('abcdef'))
    p2 = Person(name=u'Иван')

@only_for('sqlite')
class TestUnicode(unittest.TestCase):

    @db_session
    def test1(self):
        if False:
            return 10
        names = select((p.name for p in Person)).order_by(lambda : p.id)[:]
        self.assertEqual(names, ['John', u'Иван'])

    @db_session
    def test2(self):
        if False:
            i = 10
            return i + 15
        names = select((p.name.upper() for p in Person)).order_by(lambda : p.id)[:]
        self.assertEqual(names, ['JOHN', u'ИВАН'])

    @db_session
    def test3(self):
        if False:
            print('Hello World!')
        names = select((p.name.lower() for p in Person)).order_by(lambda : p.id)[:]
        self.assertEqual(names, ['john', u'иван'])

    @db_session
    def test4(self):
        if False:
            i = 10
            return i + 15
        ages = db.select('select py_upper(age) from person')
        self.assertEqual(ages, ['20', None])

    @db_session
    def test5(self):
        if False:
            while True:
                i = 10
        ages = db.select('select py_lower(age) from person')
        self.assertEqual(ages, ['20', None])

    @db_session
    def test6(self):
        if False:
            print('Hello World!')
        ages = db.select('select py_upper(image) from person')
        self.assertEqual(ages, [u'ABCDEF', None])

    @db_session
    def test7(self):
        if False:
            while True:
                i = 10
        ages = db.select('select py_lower(image) from person')
        self.assertEqual(ages, [u'abcdef', None])
if __name__ == '__main__':
    unittest.main()
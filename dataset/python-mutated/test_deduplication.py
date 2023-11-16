from io import StringIO
from pony.orm import *
from pony.orm.tests import setup_database, teardown_database
import unittest
db = Database()

class A(db.Entity):
    id = PrimaryKey(int)
    x = Required(bool)
    y = Required(float)

class TestDeduplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        setup_database(db)
        with db_session:
            a1 = A(id=1, x=False, y=3.0)
            a2 = A(id=2, x=True, y=4.0)
            a3 = A(id=3, x=False, y=1.0)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        teardown_database(db)

    @db_session
    def test_1(self):
        if False:
            i = 10
            return i + 15
        a2 = A.get(id=2)
        a1 = A.get(id=1)
        self.assertIs(a1.id, 1)

    @db_session
    def test_2(self):
        if False:
            return 10
        a3 = A.get(id=3)
        a1 = A.get(id=1)
        self.assertIs(a1.id, 1)

    @db_session
    def test_3(self):
        if False:
            return 10
        q = A.select().order_by(-1)
        stream = StringIO()
        q.show(stream=stream)
        s = stream.getvalue()
        self.assertEqual(s, 'id|x    |y  \n--+-----+---\n3 |False|1.0\n2 |True |4.0\n1 |False|3.0\n')
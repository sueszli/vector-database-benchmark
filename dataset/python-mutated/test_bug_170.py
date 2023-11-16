import unittest
from pony import orm
from pony.orm.tests import setup_database, teardown_database
db = orm.Database()

class Person(db.Entity):
    id = orm.PrimaryKey(int, auto=True)
    name = orm.Required(str)
    orm.composite_key(id, name)

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        setup_database(db)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        teardown_database(db)

    def test_1(self):
        if False:
            return 10
        table = db.schema.tables[Person._table_]
        pk_column = table.column_dict[Person.id.column]
        self.assertTrue(pk_column.is_pk)
        with orm.db_session:
            p1 = Person(name='John')
            p2 = Person(name='Mike')
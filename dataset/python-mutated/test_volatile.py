import sys, unittest
from pony.orm import *
from pony.orm.tests.testutils import *
from pony.orm.tests import setup_database, teardown_database

class TestVolatile1(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        db = self.db = Database()

        class Item(self.db.Entity):
            name = Required(str)
            index = Required(int, volatile=True)
        setup_database(db)
        with db_session:
            Item(id=1, name='A', index=1)
            Item(id=2, name='B', index=2)
            Item(id=3, name='C', index=3)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        teardown_database(self.db)

    @db_session
    def test_1(self):
        if False:
            i = 10
            return i + 15
        db = self.db
        Item = db.Item
        db.execute('update "item" set "index" = "index" + 1')
        items = Item.select(lambda item: item.index > 0).order_by(Item.id)[:]
        (a, b, c) = items
        self.assertEqual(a.index, 2)
        self.assertEqual(b.index, 3)
        self.assertEqual(c.index, 4)
        c.index = 1
        items = Item.select()[:]
        self.assertEqual(c.index, 1)
        self.assertEqual(a.index, 2)
        self.assertEqual(b.index, 3)

    @db_session
    def test_2(self):
        if False:
            return 10
        Item = self.db.Item
        item = Item[1]
        item.name = 'X'
        item.flush()
        self.assertEqual(item.index, 1)

class TestVolatile2(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        db = self.db = Database()

        class Group(db.Entity):
            number = PrimaryKey(int)
            students = Set('Student', volatile=True)

        class Student(db.Entity):
            id = PrimaryKey(int)
            name = Required(str)
            group = Required('Group')
            courses = Set('Course')

        class Course(db.Entity):
            id = PrimaryKey(int)
            name = Required(str)
            students = Set('Student', volatile=True)
        setup_database(db)
        with db_session:
            g1 = Group(number=123)
            s1 = Student(id=1, name='A', group=g1)
            s2 = Student(id=2, name='B', group=g1)
            c1 = Course(id=1, name='C1', students=[s1, s2])
            c2 = Course(id=2, name='C1', students=[s1])
        self.Group = Group
        self.Student = Student
        self.Course = Course

    def tearDown(self):
        if False:
            while True:
                i = 10
        teardown_database(self.db)

    def test_1(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.Group.students.is_volatile)
        self.assertTrue(self.Student.group.is_volatile)
        self.assertTrue(self.Student.courses.is_volatile)
        self.assertTrue(self.Course.students.is_volatile)

    def test_2(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            g1 = self.Group[123]
            students = set((s.id for s in g1.students))
            self.assertEqual(students, {1, 2})
            self.db.execute("insert into student values(3, 'C', 123)")
            g1.students.load()
            students = set((s.id for s in g1.students))
            self.assertEqual(students, {1, 2, 3})

    def test_3(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            g1 = self.Group[123]
            students = set((s.id for s in g1.students))
            self.assertEqual(students, {1, 2})
            self.db.execute("insert into student values(3, 'C', 123)")
            s3 = self.Student[3]
            students = set((s.id for s in g1.students))
            self.assertEqual(students, {1, 2, 3})

    def test_4(self):
        if False:
            return 10
        with db_session:
            c1 = self.Course[1]
            students = set((s.id for s in c1.students))
            self.assertEqual(students, {1, 2})
            self.db.execute("insert into student values(3, 'C', 123)")
            attr = self.Student.courses
            self.db.execute('insert into %s values(1, 3)' % attr.table)
            c1.students.load()
            students = set((s.id for s in c1.students))
            self.assertEqual(students, {1, 2, 3})
if __name__ == '__main__':
    unittest.main()
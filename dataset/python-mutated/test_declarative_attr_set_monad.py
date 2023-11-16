from __future__ import absolute_import, print_function, division
import unittest
from pony.orm.core import *
from pony.orm.tests.testutils import *
from pony.orm.tests import setup_database, teardown_database
db = Database()

class Student(db.Entity):
    name = Required(str)
    scholarship = Optional(int)
    group = Required('Group')
    marks = Set('Mark')

class Group(db.Entity):
    number = PrimaryKey(int)
    department = Required(int)
    students = Set(Student)
    subjects = Set('Subject')

class Subject(db.Entity):
    name = PrimaryKey(str)
    groups = Set(Group)
    marks = Set('Mark')

class Mark(db.Entity):
    value = Required(int)
    student = Required(Student)
    subject = Required(Subject)
    PrimaryKey(student, subject)

class TestAttrSetMonad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        setup_database(db)
        with db_session:
            g41 = Group(number=41, department=101)
            g42 = Group(number=42, department=102)
            g43 = Group(number=43, department=102)
            g44 = Group(number=44, department=102)
            s1 = Student(id=1, name='Joe', scholarship=None, group=g41)
            s2 = Student(id=2, name='Bob', scholarship=100, group=g41)
            s3 = Student(id=3, name='Beth', scholarship=500, group=g41)
            s4 = Student(id=4, name='Jon', scholarship=500, group=g42)
            s5 = Student(id=5, name='Pete', scholarship=700, group=g42)
            s6 = Student(id=6, name='Mary', scholarship=300, group=g44)
            Math = Subject(name='Math')
            Physics = Subject(name='Physics')
            History = Subject(name='History')
            g41.subjects = [Math, Physics, History]
            g42.subjects = [Math, Physics]
            g43.subjects = [Physics]
            Mark(value=5, student=s1, subject=Math)
            Mark(value=4, student=s2, subject=Physics)
            Mark(value=3, student=s2, subject=Math)
            Mark(value=2, student=s2, subject=History)
            Mark(value=1, student=s3, subject=History)
            Mark(value=2, student=s3, subject=Math)
            Mark(value=2, student=s4, subject=Math)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        teardown_database(db)

    def setUp(self):
        if False:
            return 10
        rollback()
        db_session.__enter__()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        rollback()
        db_session.__exit__()

    def test1(self):
        if False:
            while True:
                i = 10
        groups = select((g for g in Group if len(g.students) > 2))[:]
        self.assertEqual(groups, [Group[41]])

    def test2(self):
        if False:
            print('Hello World!')
        groups = set(select((g for g in Group if len(g.students.name) >= 2)))
        self.assertEqual(groups, {Group[41], Group[42]})

    def test3(self):
        if False:
            for i in range(10):
                print('nop')
        groups = select((g for g in Group if len(g.students.marks) > 2))[:]
        self.assertEqual(groups, [Group[41]])

    def test3a(self):
        if False:
            while True:
                i = 10
        groups = select((g for g in Group if len(g.students.marks) < 2))[:]
        self.assertEqual(set(groups), {Group[42], Group[43], Group[44]})

    def test4(self):
        if False:
            i = 10
            return i + 15
        groups = select((g for g in Group if max(g.students.marks.value) <= 2))[:]
        self.assertEqual(groups, [Group[42]])

    def test5(self):
        if False:
            for i in range(10):
                print('nop')
        students = select((s for s in Student if len(s.marks.subject.name) > 5))[:]
        self.assertEqual(students, [])

    def test6(self):
        if False:
            return 10
        students = set(select((s for s in Student if len(s.marks.subject) >= 2)))
        self.assertEqual(students, {Student[2], Student[3]})

    def test8(self):
        if False:
            while True:
                i = 10
        students = set(select((s for s in Student if s.group in (g for g in Group if g.department == 101))))
        self.assertEqual(students, {Student[1], Student[2], Student[3]})

    def test9(self):
        if False:
            while True:
                i = 10
        students = set(select((s for s in Student if s.group not in (g for g in Group if g.department == 101))))
        self.assertEqual(students, {Student[4], Student[5], Student[6]})

    def test10(self):
        if False:
            i = 10
            return i + 15
        students = set(select((s for s in Student if s.group in (g for g in Group if g.department == 101))))
        self.assertEqual(students, {Student[1], Student[2], Student[3]})

    def test11(self):
        if False:
            for i in range(10):
                print('nop')
        students = set(select((g for g in Group if len(g.subjects.groups.subjects) > 1)))
        self.assertEqual(students, {Group[41], Group[42], Group[43]})

    def test12(self):
        if False:
            while True:
                i = 10
        groups = set(select((g for g in Group if len(g.subjects) >= 2)))
        self.assertEqual(groups, {Group[41], Group[42]})

    def test13(self):
        if False:
            for i in range(10):
                print('nop')
        groups = set(select((g for g in Group if g.students)))
        self.assertEqual(groups, {Group[41], Group[42], Group[44]})

    def test14(self):
        if False:
            i = 10
            return i + 15
        groups = set(select((g for g in Group if not g.students)))
        self.assertEqual(groups, {Group[43]})

    def test15(self):
        if False:
            for i in range(10):
                print('nop')
        groups = set(select((g for g in Group if exists(g.students))))
        self.assertEqual(groups, {Group[41], Group[42], Group[44]})

    def test15a(self):
        if False:
            i = 10
            return i + 15
        groups = set(select((g for g in Group if not not exists(g.students))))
        self.assertEqual(groups, {Group[41], Group[42], Group[44]})

    def test16(self):
        if False:
            i = 10
            return i + 15
        groups = select((g for g in Group if not exists(g.students)))[:]
        self.assertEqual(groups, [Group[43]])

    def test17(self):
        if False:
            return 10
        groups = set(select((g for g in Group if 100 in g.students.scholarship)))
        self.assertEqual(groups, {Group[41]})

    def test18(self):
        if False:
            while True:
                i = 10
        groups = set(select((g for g in Group if 100 not in g.students.scholarship)))
        self.assertEqual(groups, {Group[42], Group[43], Group[44]})

    def test19(self):
        if False:
            return 10
        groups = set(select((g for g in Group if not not not 100 not in g.students.scholarship)))
        self.assertEqual(groups, {Group[41]})

    def test20(self):
        if False:
            while True:
                i = 10
        groups = set(select((g for g in Group if exists((s for s in Student if s.group == g and s.scholarship == 500)))))
        self.assertEqual(groups, {Group[41], Group[42]})

    def test21(self):
        if False:
            while True:
                i = 10
        groups = set(select((g for g in Group if g.department is not None)))
        self.assertEqual(groups, {Group[41], Group[42], Group[43], Group[44]})

    def test21a(self):
        if False:
            for i in range(10):
                print('nop')
        groups = set(select((g for g in Group if not g.department is not None)))
        self.assertEqual(groups, set())

    def test21b(self):
        if False:
            print('Hello World!')
        groups = set(select((g for g in Group if not not not g.department is None)))
        self.assertEqual(groups, {Group[41], Group[42], Group[43], Group[44]})

    def test22(self):
        if False:
            for i in range(10):
                print('nop')
        groups = set(select((g for g in Group if 700 in (s.scholarship for s in Student if s.group == g))))
        self.assertEqual(groups, {Group[42]})

    def test23a(self):
        if False:
            for i in range(10):
                print('nop')
        groups = set(select((g for g in Group if 700 not in g.students.scholarship)))
        self.assertEqual(groups, {Group[41], Group[43], Group[44]})

    def test23b(self):
        if False:
            while True:
                i = 10
        groups = set(select((g for g in Group if 700 not in (s.scholarship for s in Student if s.group == g))))
        self.assertEqual(groups, {Group[41], Group[43], Group[44]})

    @raises_exception(NotImplementedError)
    def test24(self):
        if False:
            i = 10
            return i + 15
        groups = set(select((g for g in Group for g2 in Group if g.students == g2.students)))

    def test25(self):
        if False:
            for i in range(10):
                print('nop')
        m1 = Mark[Student[1], Subject['Math']]
        students = set(select((s for s in Student if m1 in s.marks)))
        self.assertEqual(students, {Student[1]})

    def test26(self):
        if False:
            while True:
                i = 10
        s1 = Student[1]
        groups = set(select((g for g in Group if s1 in g.students)))
        self.assertEqual(groups, {Group[41]})

    @raises_exception(AttributeError, 'g.students.name.foo')
    def test27(self):
        if False:
            i = 10
            return i + 15
        select((g for g in Group if g.students.name.foo == 1))

    def test28(self):
        if False:
            print('Hello World!')
        groups = set(select((g for g in Group if not g.students.is_empty())))
        self.assertEqual(groups, {Group[41], Group[42], Group[44]})

    @raises_exception(NotImplementedError)
    def test29(self):
        if False:
            for i in range(10):
                print('nop')
        students = select((g.students.select(lambda s: s.scholarship > 0) for g in Group if g.department == 101))[:]

    def test30a(self):
        if False:
            i = 10
            return i + 15
        s = Student[2]
        groups = select((g for g in Group if g.department == 101 and s in g.students.select(lambda s: s.scholarship > 0)))[:]
        self.assertEqual(set(groups), {Group[41]})

    def test30b(self):
        if False:
            while True:
                i = 10
        s = Student[2]
        groups = select((g for g in Group if g.department == 101 and s in g.students.filter(lambda s: s.scholarship > 0)))[:]
        self.assertEqual(set(groups), {Group[41]})

    def test30c(self):
        if False:
            print('Hello World!')
        s = Student[2]
        groups = select((g for g in Group if g.department == 101 and s in g.students.select()))[:]
        self.assertEqual(set(groups), {Group[41]})

    def test30d(self):
        if False:
            for i in range(10):
                print('nop')
        s = Student[2]
        groups = select((g for g in Group if g.department == 101 and s in g.students.filter()))[:]
        self.assertEqual(set(groups), {Group[41]})

    def test31(self):
        if False:
            print('Hello World!')
        s = Student[2]
        groups = select((g for g in Group if g.department == 101 and g.students.exists(lambda s: s.scholarship > 0)))[:]
        self.assertEqual(set(groups), {Group[41]})
if __name__ == '__main__':
    unittest.main()
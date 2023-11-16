from __future__ import absolute_import, print_function, division
import unittest, warnings
from datetime import date
from decimal import Decimal
from itertools import count
from pony.orm.core import *
from pony.orm.tests.testutils import *
from pony.orm.tests import setup_database, teardown_database

class TestDBSession(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.db = Database()

        class X(self.db.Entity):
            a = PrimaryKey(int)
            b = Optional(int)
        self.X = X
        setup_database(self.db)
        with db_session:
            x1 = X(a=1, b=1)
            x2 = X(a=2, b=2)

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.db.provider.dialect != 'SQLite':
            teardown_database(self.db)

    @raises_exception(TypeError, 'Pass only keyword arguments to db_session or use db_session as decorator')
    def test_db_session_1(self):
        if False:
            print('Hello World!')
        db_session(1, 2, 3)

    @raises_exception(TypeError, 'Pass only keyword arguments to db_session or use db_session as decorator')
    def test_db_session_2(self):
        if False:
            print('Hello World!')
        db_session(1, 2, 3, a=10, b=20)

    def test_db_session_3(self):
        if False:
            return 10
        self.assertTrue(db_session is db_session())

    def test_db_session_4(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            with db_session:
                self.X(a=3, b=3)
        with db_session:
            self.assertEqual(count((x for x in self.X)), 3)

    def test_db_session_decorator_1(self):
        if False:
            i = 10
            return i + 15

        @db_session
        def test():
            if False:
                i = 10
                return i + 15
            self.X(a=3, b=3)
        test()
        with db_session:
            self.assertEqual(count((x for x in self.X)), 3)

    def test_db_session_decorator_2(self):
        if False:
            print('Hello World!')

        @db_session
        def test():
            if False:
                return 10
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_db_session_decorator_3(self):
        if False:
            return 10

        @db_session(allowed_exceptions=[TypeError])
        def test():
            if False:
                while True:
                    i = 10
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_db_session_decorator_4(self):
        if False:
            print('Hello World!')

        @db_session(allowed_exceptions=[ZeroDivisionError])
        def test():
            if False:
                i = 10
                return i + 15
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 3)
        else:
            self.fail()

    def test_allowed_exceptions_1(self):
        if False:
            while True:
                i = 10

        @db_session(allowed_exceptions=lambda e: isinstance(e, ZeroDivisionError))
        def test():
            if False:
                print('Hello World!')
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 3)
        else:
            self.fail()

    def test_allowed_exceptions_2(self):
        if False:
            for i in range(10):
                print('nop')

        @db_session(allowed_exceptions=lambda e: isinstance(e, TypeError))
        def test():
            if False:
                print('Hello World!')
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    @raises_exception(TypeError, "'retry' parameter of db_session must be of integer type. Got: %r" % str)
    def test_retry_1(self):
        if False:
            return 10

        @db_session(retry='foobar')
        def test():
            if False:
                print('Hello World!')
            pass

    @raises_exception(TypeError, "'retry' parameter of db_session must not be negative. Got: -1")
    def test_retry_2(self):
        if False:
            i = 10
            return i + 15

        @db_session(retry=-1)
        def test():
            if False:
                print('Hello World!')
            pass

    def test_retry_3(self):
        if False:
            while True:
                i = 10
        counter = count()

        @db_session(retry_exceptions=[ZeroDivisionError])
        def test():
            if False:
                for i in range(10):
                    print('nop')
            next(counter)
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.assertEqual(next(counter), 1)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_retry_4(self):
        if False:
            return 10
        counter = count()

        @db_session(retry=1, retry_exceptions=[ZeroDivisionError])
        def test():
            if False:
                return 10
            next(counter)
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.assertEqual(next(counter), 2)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_retry_5(self):
        if False:
            i = 10
            return i + 15
        counter = count()

        @db_session(retry=5, retry_exceptions=[ZeroDivisionError])
        def test():
            if False:
                i = 10
                return i + 15
            next(counter)
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.assertEqual(next(counter), 6)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_retry_6(self):
        if False:
            while True:
                i = 10
        counter = count()

        @db_session(retry=3, retry_exceptions=[TypeError])
        def test():
            if False:
                while True:
                    i = 10
            next(counter)
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.assertEqual(next(counter), 1)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_retry_7(self):
        if False:
            print('Hello World!')
        counter = count()

        @db_session(retry=5, retry_exceptions=[ZeroDivisionError])
        def test():
            if False:
                print('Hello World!')
            i = next(counter)
            self.X(a=3, b=3)
            if i < 2:
                1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.fail()
        else:
            self.assertEqual(next(counter), 3)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 3)

    @raises_exception(TypeError, 'The same exception ZeroDivisionError cannot be specified in both allowed and retry exception lists simultaneously')
    def test_retry_8(self):
        if False:
            for i in range(10):
                print('nop')

        @db_session(retry=3, retry_exceptions=[ZeroDivisionError], allowed_exceptions=[ZeroDivisionError])
        def test():
            if False:
                return 10
            pass

    def test_retry_9(self):
        if False:
            return 10
        counter = count()

        @db_session(retry=3, retry_exceptions=lambda e: isinstance(e, ZeroDivisionError))
        def test():
            if False:
                while True:
                    i = 10
            i = next(counter)
            self.X(a=3, b=3)
            1 / 0
        try:
            test()
        except ZeroDivisionError:
            self.assertEqual(next(counter), 4)
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_retry_10(self):
        if False:
            for i in range(10):
                print('nop')
        retries = count()

        @db_session(retry=3)
        def test():
            if False:
                while True:
                    i = 10
            next(retries)
            self.X(a=1, b=1)
        try:
            test()
        except TransactionIntegrityError:
            self.assertEqual(next(retries), 4)
        else:
            self.fail()

    @raises_exception(PonyRuntimeWarning, '@db_session decorator with `retry=3` option is ignored for test() function because it is called inside another db_session')
    def test_retry_11(self):
        if False:
            while True:
                i = 10

        @db_session(retry=3)
        def test():
            if False:
                for i in range(10):
                    print('nop')
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('error', PonyRuntimeWarning)
            with db_session:
                test()

    def test_db_session_manager_1(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            self.X(a=3, b=3)
        with db_session:
            self.assertEqual(count((x for x in self.X)), 3)

    @raises_exception(TypeError, "@db_session can accept 'retry' parameter only when used as decorator and not as context manager")
    def test_db_session_manager_2(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session(retry=3):
            self.X(a=3, b=3)

    def test_db_session_manager_3(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            with db_session(allowed_exceptions=[TypeError]):
                self.X(a=3, b=3)
                1 / 0
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 2)
        else:
            self.fail()

    def test_db_session_manager_4(self):
        if False:
            while True:
                i = 10
        try:
            with db_session(allowed_exceptions=[ZeroDivisionError]):
                self.X(a=3, b=3)
                1 / 0
        except ZeroDivisionError:
            with db_session:
                self.assertEqual(count((x for x in self.X)), 3)
        else:
            self.fail()

    def test_db_session_ddl_1(self):
        if False:
            print('Hello World!')
        with db_session(ddl=True):
            pass

    def test_db_session_ddl_1a(self):
        if False:
            i = 10
            return i + 15
        with db_session(ddl=True):
            with db_session(ddl=True):
                pass

    def test_db_session_ddl_1b(self):
        if False:
            while True:
                i = 10
        with db_session(ddl=True):
            with db_session:
                pass

    @raises_exception(TransactionError, 'Cannot start ddl transaction inside non-ddl transaction')
    def test_db_session_ddl_1c(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            with db_session(ddl=True):
                pass

    @raises_exception(TransactionError, '@db_session-decorated test() function with `ddl` option cannot be called inside of another db_session')
    def test_db_session_ddl_2(self):
        if False:
            for i in range(10):
                print('nop')

        @db_session(ddl=True)
        def test():
            if False:
                print('Hello World!')
            pass
        with db_session:
            test()

    def test_db_session_ddl_3(self):
        if False:
            while True:
                i = 10

        @db_session(ddl=True)
        def test():
            if False:
                print('Hello World!')
            pass
        test()

    @raises_exception(ZeroDivisionError)
    def test_db_session_exceptions_1(self):
        if False:
            while True:
                i = 10

        def before_insert(self):
            if False:
                print('Hello World!')
            1 / 0
        self.X.before_insert = before_insert
        with db_session:
            self.X(a=3, b=3)

    @raises_exception(ZeroDivisionError)
    def test_db_session_exceptions_2(self):
        if False:
            return 10

        def before_insert(self):
            if False:
                for i in range(10):
                    print('nop')
            1 / 0
        self.X.before_insert = before_insert
        with db_session:
            self.X(a=3, b=3)
            commit()

    @raises_exception(ZeroDivisionError)
    def test_db_session_exceptions_3(self):
        if False:
            i = 10
            return i + 15

        def before_insert(self):
            if False:
                i = 10
                return i + 15
            1 / 0
        self.X.before_insert = before_insert
        with db_session:
            self.X(a=3, b=3)
            db.commit()

    @raises_exception(ZeroDivisionError)
    def test_db_session_exceptions_4(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            connection = self.db.get_connection()
            connection.close()
            1 / 0
db = Database()

class Group(db.Entity):
    id = PrimaryKey(int)
    major = Required(str)
    students = Set('Student')

class Student(db.Entity):
    name = Required(str)
    picture = Optional(buffer, lazy=True)
    group = Required('Group')

class TestDBSessionScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        setup_database(db)
        with db_session:
            g1 = Group(id=1, major='Math')
            g2 = Group(id=2, major='Physics')
            s1 = Student(id=1, name='S1', group=g1)
            s2 = Student(id=2, name='S2', group=g1)
            s3 = Student(id=3, name='S3', group=g2)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        teardown_database(db)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        rollback()

    def tearDown(self):
        if False:
            return 10
        rollback()

    def test1(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            s1 = Student[1]
        name = s1.name

    @raises_exception(DatabaseSessionIsOver, 'Cannot load attribute Student[1].picture: the database session is over')
    def test2(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            s1 = Student[1]
        picture = s1.picture

    @raises_exception(DatabaseSessionIsOver, 'Cannot load attribute Group[1].major: the database session is over')
    def test3(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            s1 = Student[1]
        group_id = s1.group.id
        major = s1.group.major

    @raises_exception(DatabaseSessionIsOver, 'Cannot assign new value to Student[1].name: the database session is over')
    def test4(self):
        if False:
            print('Hello World!')
        with db_session:
            s1 = Student[1]
        s1.name = 'New name'

    def test5(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            g1 = Group[1]
        self.assertEqual(str(g1.students), 'StudentSet([...])')

    @raises_exception(DatabaseSessionIsOver, 'Cannot load collection Group[1].students: the database session is over')
    def test6(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session:
            g1 = Group[1]
        l = len(g1.students)

    @raises_exception(DatabaseSessionIsOver, 'Cannot change collection Group[1].students: the database session is over')
    def test7(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            s1 = Student[1]
            g1 = Group[1]
        g1.students.remove(s1)

    @raises_exception(DatabaseSessionIsOver, 'Cannot change collection Group[1].students: the database session is over')
    def test8(self):
        if False:
            while True:
                i = 10
        with db_session:
            g2_students = Group[2].students
            g1 = Group[1]
        g1.students = g2_students

    @raises_exception(DatabaseSessionIsOver, 'Cannot change collection Group[1].students: the database session is over')
    def test9(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            s3 = Student[3]
            g1 = Group[1]
        g1.students.add(s3)

    @raises_exception(DatabaseSessionIsOver, 'Cannot change collection Group[1].students: the database session is over')
    def test10(self):
        if False:
            i = 10
            return i + 15
        with db_session:
            g1 = Group[1]
        g1.students.clear()

    @raises_exception(DatabaseSessionIsOver, 'Cannot delete object Student[1]: the database session is over')
    def test11(self):
        if False:
            while True:
                i = 10
        with db_session:
            s1 = Student[1]
        s1.delete()

    @raises_exception(DatabaseSessionIsOver, 'Cannot change object Student[1]: the database session is over')
    def test12(self):
        if False:
            print('Hello World!')
        with db_session:
            s1 = Student[1]
        s1.set(name='New name')

    def test_db_session_strict_1(self):
        if False:
            return 10
        with db_session(strict=True):
            s1 = Student[1]

    @raises_exception(DatabaseSessionIsOver, 'Cannot read value of Student[1].name: the database session is over')
    def test_db_session_strict_2(self):
        if False:
            for i in range(10):
                print('nop')
        with db_session(strict=True):
            s1 = Student[1]
        name = s1.name
if __name__ == '__main__':
    unittest.main()
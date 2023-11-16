import unittest
from uvloop import _testbase as tb

class TestBaseTest(unittest.TestCase):

    def test_duplicate_methods(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'duplicate test Foo.test_a'):

            class Foo(tb.BaseTestCase):

                def test_a(self):
                    if False:
                        return 10
                    pass

                def test_b(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                def test_a(self):
                    if False:
                        i = 10
                        return i + 15
                    pass

    def test_duplicate_methods_parent_1(self):
        if False:
            for i in range(10):
                print('nop')

        class FooBase:

            def test_a(self):
                if False:
                    print('Hello World!')
                pass
        with self.assertRaisesRegex(RuntimeError, 'duplicate test Foo.test_a.*defined in FooBase'):

            class Foo(FooBase, tb.BaseTestCase):

                def test_b(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                def test_a(self):
                    if False:
                        i = 10
                        return i + 15
                    pass

    def test_duplicate_methods_parent_2(self):
        if False:
            return 10

        class FooBase(tb.BaseTestCase):

            def test_a(self):
                if False:
                    while True:
                        i = 10
                pass
        with self.assertRaisesRegex(RuntimeError, 'duplicate test Foo.test_a.*defined in FooBase'):

            class Foo(FooBase):

                def test_b(self):
                    if False:
                        print('Hello World!')
                    pass

                def test_a(self):
                    if False:
                        print('Hello World!')
                    pass
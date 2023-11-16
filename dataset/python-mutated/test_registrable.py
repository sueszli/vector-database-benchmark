from unittest import TestCase
from snips_nlu.common.registrable import Registrable
from snips_nlu.exceptions import NotRegisteredError, AlreadyRegisteredError

class TestRegistrable(TestCase):

    def test_should_register_subclass(self):
        if False:
            i = 10
            return i + 15

        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register('first_subclass')
        class MyFirstSubclass(MyBaseClass):
            pass

        @MyBaseClass.register('second_subclass')
        class MySecondSubclass(MyBaseClass):
            pass
        my_subclass = MyBaseClass.by_name('second_subclass')
        self.assertEqual(MySecondSubclass, my_subclass)

    def test_should_raise_when_not_registered(self):
        if False:
            for i in range(10):
                print('nop')

        class MyBaseClass(Registrable):
            pass
        with self.assertRaises(NotRegisteredError):
            MyBaseClass.by_name('my_unregistered_subclass')

    def test_should_raise_when_already_registered(self):
        if False:
            for i in range(10):
                print('nop')

        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register('my_duplicated_subclass')
        class MySubclass(MyBaseClass):
            pass
        with self.assertRaises(AlreadyRegisteredError):

            @MyBaseClass.register('my_duplicated_subclass')
            class MySecondSubclass(MyBaseClass):
                pass

    def test_should_override_already_registered_subclass(self):
        if False:
            print('Hello World!')

        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register('my_subclass')
        class MyOverridenSubclass(MyBaseClass):
            pass

        @MyBaseClass.register('my_subclass', override=True)
        class MySubclass(MyBaseClass):
            pass
        subclass = MyBaseClass.by_name('my_subclass')
        self.assertEqual(MySubclass, subclass)
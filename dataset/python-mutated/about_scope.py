from runner.koan import *
from . import jims
from . import joes
counter = 0

class AboutScope(Koan):

    def test_dog_is_not_available_in_the_current_scope(self):
        if False:
            print('Hello World!')
        with self.assertRaises(___):
            fido = Dog()

    def test_you_can_reference_nested_classes_using_the_scope_operator(self):
        if False:
            while True:
                i = 10
        fido = jims.Dog()
        rover = joes.Dog()
        self.assertEqual(__, fido.identify())
        self.assertEqual(__, rover.identify())
        self.assertEqual(__, type(fido) == type(rover))
        self.assertEqual(__, jims.Dog == joes.Dog)

    class str:
        pass

    def test_bare_bones_class_names_do_not_assume_the_current_scope(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, AboutScope.str == str)

    def test_nested_string_is_not_the_same_as_the_system_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, self.str == type('HI'))

    def test_str_without_self_prefix_stays_in_the_global_scope(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, str == type('HI'))
    PI = 3.1416

    def test_constants_are_defined_with_an_initial_uppercase_letter(self):
        if False:
            while True:
                i = 10
        self.assertAlmostEqual(_____, self.PI)

    def test_constants_are_assumed_by_convention_only(self):
        if False:
            for i in range(10):
                print('nop')
        self.PI = 'rhubarb'
        self.assertEqual(_____, self.PI)

    def increment_using_local_counter(self, counter):
        if False:
            i = 10
            return i + 15
        counter = counter + 1

    def increment_using_global_counter(self):
        if False:
            while True:
                i = 10
        global counter
        counter = counter + 1

    def test_incrementing_with_local_counter(self):
        if False:
            for i in range(10):
                print('nop')
        global counter
        start = counter
        self.increment_using_local_counter(start)
        self.assertEqual(__, counter == start + 1)

    def test_incrementing_with_global_counter(self):
        if False:
            for i in range(10):
                print('nop')
        global counter
        start = counter
        self.increment_using_global_counter()
        self.assertEqual(__, counter == start + 1)

    def local_access(self):
        if False:
            return 10
        stuff = 'eels'

        def from_the_league():
            if False:
                print('Hello World!')
            stuff = 'this is a local shop for local people'
            return stuff
        return from_the_league()

    def nonlocal_access(self):
        if False:
            print('Hello World!')
        stuff = 'eels'

        def from_the_boosh():
            if False:
                i = 10
                return i + 15
            nonlocal stuff
            return stuff
        return from_the_boosh()

    def test_getting_something_locally(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.local_access())

    def test_getting_something_nonlocally(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, self.nonlocal_access())
    global deadly_bingo
    deadly_bingo = [4, 8, 15, 16, 23, 42]

    def test_global_attributes_can_be_created_in_the_middle_of_a_class(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, deadly_bingo[5])
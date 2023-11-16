from runner.koan import *

class AboutMonkeyPatching(Koan):

    class Dog:

        def bark(self):
            if False:
                while True:
                    i = 10
            return 'WOOF'

    def test_as_defined_dogs_do_bark(self):
        if False:
            for i in range(10):
                print('nop')
        fido = self.Dog()
        self.assertEqual(__, fido.bark())

    def test_after_patching_dogs_can_both_wag_and_bark(self):
        if False:
            i = 10
            return i + 15

        def wag(self):
            if False:
                i = 10
                return i + 15
            return 'HAPPY'
        self.Dog.wag = wag
        fido = self.Dog()
        self.assertEqual(__, fido.wag())
        self.assertEqual(__, fido.bark())

    def test_most_built_in_classes_cannot_be_monkey_patched(self):
        if False:
            while True:
                i = 10
        try:
            int.is_even = lambda self: self % 2 == 0
        except Exception as ex:
            err_msg = ex.args[0]
        self.assertRegex(err_msg, __)

    class MyInt(int):
        pass

    def test_subclasses_of_built_in_classes_can_be_be_monkey_patched(self):
        if False:
            while True:
                i = 10
        self.MyInt.is_even = lambda self: self % 2 == 0
        self.assertEqual(__, self.MyInt(1).is_even())
        self.assertEqual(__, self.MyInt(2).is_even())
from runner.koan import *

class AboutInheritance(Koan):

    class Dog:

        def __init__(self, name):
            if False:
                return 10
            self._name = name

        @property
        def name(self):
            if False:
                while True:
                    i = 10
            return self._name

        def bark(self):
            if False:
                print('Hello World!')
            return 'WOOF'

    class Chihuahua(Dog):

        def wag(self):
            if False:
                return 10
            return 'happy'

        def bark(self):
            if False:
                return 10
            return 'yip'

    def test_subclasses_have_the_parent_as_an_ancestor(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, issubclass(self.Chihuahua, self.Dog))

    def test_all_classes_in_python_3_ultimately_inherit_from_object_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, issubclass(self.Chihuahua, object))

    def test_instances_inherit_behavior_from_parent_class(self):
        if False:
            return 10
        chico = self.Chihuahua('Chico')
        self.assertEqual(__, chico.name)

    def test_subclasses_add_new_behavior(self):
        if False:
            return 10
        chico = self.Chihuahua('Chico')
        self.assertEqual(__, chico.wag())
        fido = self.Dog('Fido')
        with self.assertRaises(___):
            fido.wag()

    def test_subclasses_can_modify_existing_behavior(self):
        if False:
            print('Hello World!')
        chico = self.Chihuahua('Chico')
        self.assertEqual(__, chico.bark())
        fido = self.Dog('Fido')
        self.assertEqual(__, fido.bark())

    class BullDog(Dog):

        def bark(self):
            if False:
                print('Hello World!')
            return super().bark() + ', GRR'

    def test_subclasses_can_invoke_parent_behavior_via_super(self):
        if False:
            print('Hello World!')
        ralph = self.BullDog('Ralph')
        self.assertEqual(__, ralph.bark())

    class GreatDane(Dog):

        def growl(self):
            if False:
                while True:
                    i = 10
            return super().bark() + ', GROWL'

    def test_super_works_across_methods(self):
        if False:
            print('Hello World!')
        george = self.GreatDane('George')
        self.assertEqual(__, george.growl())

    class Pug(Dog):

        def __init__(self, name):
            if False:
                return 10
            pass

    class Greyhound(Dog):

        def __init__(self, name):
            if False:
                i = 10
                return i + 15
            super().__init__(name)

    def test_base_init_does_not_get_called_automatically(self):
        if False:
            i = 10
            return i + 15
        snoopy = self.Pug('Snoopy')
        with self.assertRaises(___):
            name = snoopy.name

    def test_base_init_has_to_be_called_explicitly(self):
        if False:
            for i in range(10):
                print('nop')
        boxer = self.Greyhound('Boxer')
        self.assertEqual(__, boxer.name)
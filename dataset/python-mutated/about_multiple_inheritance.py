from runner.koan import *

class AboutMultipleInheritance(Koan):

    class Nameable:

        def __init__(self):
            if False:
                print('Hello World!')
            self._name = None

        def set_name(self, new_name):
            if False:
                while True:
                    i = 10
            self._name = new_name

        def here(self):
            if False:
                return 10
            return 'In Nameable class'

    class Animal:

        def legs(self):
            if False:
                return 10
            return 4

        def can_climb_walls(self):
            if False:
                while True:
                    i = 10
            return False

        def here(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'In Animal class'

    class Pig(Animal):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self._name = 'Jasper'

        @property
        def name(self):
            if False:
                return 10
            return self._name

        def speak(self):
            if False:
                print('Hello World!')
            return 'OINK'

        def color(self):
            if False:
                while True:
                    i = 10
            return 'pink'

        def here(self):
            if False:
                return 10
            return 'In Pig class'

    class Spider(Animal):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self._name = 'Boris'

        def can_climb_walls(self):
            if False:
                print('Hello World!')
            return True

        def legs(self):
            if False:
                for i in range(10):
                    print('nop')
            return 8

        def color(self):
            if False:
                i = 10
                return i + 15
            return 'black'

        def here(self):
            if False:
                while True:
                    i = 10
            return 'In Spider class'

    class Spiderpig(Pig, Spider, Nameable):

        def __init__(self):
            if False:
                print('Hello World!')
            super(AboutMultipleInheritance.Pig, self).__init__()
            super(AboutMultipleInheritance.Nameable, self).__init__()
            self._name = 'Jeff'

        def speak(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'This looks like a job for Spiderpig!'

        def here(self):
            if False:
                print('Hello World!')
            return 'In Spiderpig class'

    def test_normal_methods_are_available_in_the_object(self):
        if False:
            i = 10
            return i + 15
        jeff = self.Spiderpig()
        self.assertRegex(jeff.speak(), __)

    def test_base_class_methods_are_also_available_in_the_object(self):
        if False:
            i = 10
            return i + 15
        jeff = self.Spiderpig()
        try:
            jeff.set_name('Rover')
        except:
            self.fail('This should not happen')
        self.assertEqual(__, jeff.can_climb_walls())

    def test_base_class_methods_can_affect_instance_variables_in_the_object(self):
        if False:
            while True:
                i = 10
        jeff = self.Spiderpig()
        self.assertEqual(__, jeff.name)
        jeff.set_name('Rover')
        self.assertEqual(__, jeff.name)

    def test_left_hand_side_inheritance_tends_to_be_higher_priority(self):
        if False:
            for i in range(10):
                print('nop')
        jeff = self.Spiderpig()
        self.assertEqual(__, jeff.color())

    def test_super_class_methods_are_higher_priority_than_super_super_classes(self):
        if False:
            return 10
        jeff = self.Spiderpig()
        self.assertEqual(__, jeff.legs())

    def test_we_can_inspect_the_method_resolution_order(self):
        if False:
            return 10
        mro = type(self.Spiderpig()).mro()
        self.assertEqual('Spiderpig', mro[0].__name__)
        self.assertEqual('Pig', mro[1].__name__)
        self.assertEqual(__, mro[2].__name__)
        self.assertEqual(__, mro[3].__name__)
        self.assertEqual(__, mro[4].__name__)
        self.assertEqual(__, mro[5].__name__)

    def test_confirm_the_mro_controls_the_calling_order(self):
        if False:
            print('Hello World!')
        jeff = self.Spiderpig()
        self.assertRegex(jeff.here(), 'Spiderpig')
        next = super(AboutMultipleInheritance.Spiderpig, jeff)
        self.assertRegex(next.here(), 'Pig')
        next = super(AboutMultipleInheritance.Pig, jeff)
        self.assertRegex(next.here(), __)
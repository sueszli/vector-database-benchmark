from runner.koan import *

def function():
    if False:
        while True:
            i = 10
    return 'pineapple'

def function2():
    if False:
        return 10
    return 'tractor'

class Class:

    def method(self):
        if False:
            print('Hello World!')
        return 'parrot'

class AboutMethodBindings(Koan):

    def test_methods_are_bound_to_an_object(self):
        if False:
            return 10
        obj = Class()
        self.assertEqual(__, obj.method.__self__ == obj)

    def test_methods_are_also_bound_to_a_function(self):
        if False:
            for i in range(10):
                print('nop')
        obj = Class()
        self.assertEqual(__, obj.method())
        self.assertEqual(__, obj.method.__func__(obj))

    def test_functions_have_attributes(self):
        if False:
            return 10
        obj = Class()
        self.assertEqual(__, len(dir(function)))
        self.assertEqual(__, dir(function) == dir(obj.method.__func__))

    def test_methods_have_different_attributes(self):
        if False:
            print('Hello World!')
        obj = Class()
        self.assertEqual(__, len(dir(obj.method)))

    def test_setting_attributes_on_an_unbound_function(self):
        if False:
            return 10
        function.cherries = 3
        self.assertEqual(__, function.cherries)

    def test_setting_attributes_on_a_bound_method_directly(self):
        if False:
            return 10
        obj = Class()
        with self.assertRaises(___):
            obj.method.cherries = 3

    def test_setting_attributes_on_methods_by_accessing_the_inner_function(self):
        if False:
            i = 10
            return i + 15
        obj = Class()
        obj.method.__func__.cherries = 3
        self.assertEqual(__, obj.method.cherries)

    def test_functions_can_have_inner_functions(self):
        if False:
            return 10
        function2.get_fruit = function
        self.assertEqual(__, function2.get_fruit())

    def test_inner_functions_are_unbound(self):
        if False:
            for i in range(10):
                print('nop')
        function2.get_fruit = function
        with self.assertRaises(___):
            cls = function2.get_fruit.__self__

    class BoundClass:

        def __get__(self, obj, cls):
            if False:
                i = 10
                return i + 15
            return (self, obj, cls)
    binding = BoundClass()

    def test_get_descriptor_resolves_attribute_binding(self):
        if False:
            return 10
        (bound_obj, binding_owner, owner_type) = self.binding
        self.assertEqual(__, bound_obj.__class__.__name__)
        self.assertEqual(__, binding_owner.__class__.__name__)
        self.assertEqual(AboutMethodBindings, owner_type)

    class SuperColor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.choice = None

        def __set__(self, obj, val):
            if False:
                i = 10
                return i + 15
            self.choice = val
    color = SuperColor()

    def test_set_descriptor_changes_behavior_of_attribute_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(None, self.color.choice)
        self.color = 'purple'
        self.assertEqual(__, self.color.choice)
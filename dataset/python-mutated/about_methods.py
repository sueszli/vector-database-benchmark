from runner.koan import *

def my_global_function(a, b):
    if False:
        print('Hello World!')
    return a + b

class AboutMethods(Koan):

    def test_calling_a_global_function(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, my_global_function(2, 3))

    def test_calling_functions_with_wrong_number_of_arguments(self):
        if False:
            print('Hello World!')
        try:
            my_global_function()
        except TypeError as exception:
            msg = exception.args[0]
        self.assertRegex(msg, 'my_global_function\\(\\) missing 2 required positional arguments')
        try:
            my_global_function(1, 2, 3)
        except Exception as e:
            msg = e.args[0]
        self.assertRegex(msg, __)

    def pointless_method(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        sum = a + b

    def test_which_does_not_return_anything(self):
        if False:
            return 10
        self.assertEqual(__, self.pointless_method(1, 2))

    def method_with_defaults(self, a, b='default_value'):
        if False:
            i = 10
            return i + 15
        return [a, b]

    def test_calling_with_default_values(self):
        if False:
            return 10
        self.assertEqual(__, self.method_with_defaults(1))
        self.assertEqual(__, self.method_with_defaults(1, 2))

    def method_with_var_args(self, *args):
        if False:
            i = 10
            return i + 15
        return args

    def test_calling_with_variable_arguments(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.method_with_var_args())
        self.assertEqual(('one',), self.method_with_var_args('one'))
        self.assertEqual(__, self.method_with_var_args('one', 'two'))

    def function_with_the_same_name(self, a, b):
        if False:
            i = 10
            return i + 15
        return a + b

    def test_functions_without_self_arg_are_global_functions(self):
        if False:
            for i in range(10):
                print('nop')

        def function_with_the_same_name(a, b):
            if False:
                return 10
            return a * b
        self.assertEqual(__, function_with_the_same_name(3, 4))

    def test_calling_methods_in_same_class_with_explicit_receiver(self):
        if False:
            print('Hello World!')

        def function_with_the_same_name(a, b):
            if False:
                while True:
                    i = 10
            return a * b
        self.assertEqual(__, self.function_with_the_same_name(3, 4))

    def another_method_with_the_same_name(self):
        if False:
            return 10
        return 10
    link_to_overlapped_method = another_method_with_the_same_name

    def another_method_with_the_same_name(self):
        if False:
            return 10
        return 42

    def test_that_old_methods_are_hidden_by_redefinitions(self):
        if False:
            return 10
        self.assertEqual(__, self.another_method_with_the_same_name())

    def test_that_overlapped_method_is_still_there(self):
        if False:
            return 10
        self.assertEqual(__, self.link_to_overlapped_method())

    def empty_method(self):
        if False:
            return 10
        pass

    def test_methods_that_do_nothing_need_to_use_pass_as_a_filler(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.empty_method())

    def test_pass_does_nothing_at_all(self):
        if False:
            return 10
        'You'
        'shall'
        'not'
        pass
        self.assertEqual(____, 'Still got to this line' != None)

    def one_line_method(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Madagascar'

    def test_no_indentation_required_for_one_line_statement_bodies(self):
        if False:
            while True:
                i = 10
        self.assertEqual(__, self.one_line_method())

    def method_with_documentation(self):
        if False:
            print('Hello World!')
        'A string placed at the beginning of a function is used for documentation'
        return 'ok'

    def test_the_documentation_can_be_viewed_with_the_doc_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRegex(self.method_with_documentation.__doc__, __)

    class Dog:

        def name(self):
            if False:
                print('Hello World!')
            return 'Fido'

        def _tail(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'wagging'

        def __password(self):
            if False:
                print('Hello World!')
            return 'password'

    def test_calling_methods_in_other_objects(self):
        if False:
            return 10
        rover = self.Dog()
        self.assertEqual(__, rover.name())

    def test_private_access_is_implied_but_not_enforced(self):
        if False:
            i = 10
            return i + 15
        rover = self.Dog()
        self.assertEqual(__, rover._tail())

    def test_attributes_with_double_underscore_prefixes_are_subject_to_name_mangling(self):
        if False:
            i = 10
            return i + 15
        rover = self.Dog()
        with self.assertRaises(___):
            password = rover.__password()
        self.assertEqual(__, rover._Dog__password())
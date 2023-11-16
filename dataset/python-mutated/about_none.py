from runner.koan import *

class AboutNone(Koan):

    def test_none_is_an_object(self):
        if False:
            i = 10
            return i + 15
        'Unlike NULL in a lot of languages'
        self.assertEqual(__, isinstance(None, object))

    def test_none_is_universal(self):
        if False:
            return 10
        'There is only one None'
        self.assertEqual(____, None is None)

    def test_what_exception_do_you_get_when_calling_nonexistent_methods(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        What is the Exception that is thrown when you call a method that does\n        not exist?\n\n        Hint: launch python command console and try the code in the block below.\n\n        Don't worry about what 'try' and 'except' do, we'll talk about this later\n        "
        try:
            None.some_method_none_does_not_know_about()
        except Exception as ex:
            ex2 = ex
        self.assertEqual(__, ex2.__class__)
        self.assertRegex(ex2.args[0], __)

    def test_none_is_distinct(self):
        if False:
            print('Hello World!')
        '\n        None is distinct from other things which are False.\n        '
        self.assertEqual(__, None is not 0)
        self.assertEqual(__, None is not False)
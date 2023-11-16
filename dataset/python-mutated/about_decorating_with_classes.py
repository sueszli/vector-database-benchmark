from runner.koan import *
import functools

class AboutDecoratingWithClasses(Koan):

    def maximum(self, a, b):
        if False:
            print('Hello World!')
        if a > b:
            return a
        else:
            return b

    def test_partial_that_wrappers_no_args(self):
        if False:
            return 10
        '\n        Before we can understand this type of decorator we need to consider\n        the partial.\n        '
        max = functools.partial(self.maximum)
        self.assertEqual(__, max(7, 23))
        self.assertEqual(__, max(10, -10))

    def test_partial_that_wrappers_first_arg(self):
        if False:
            print('Hello World!')
        max0 = functools.partial(self.maximum, 0)
        self.assertEqual(__, max0(-4))
        self.assertEqual(__, max0(5))

    def test_partial_that_wrappers_all_args(self):
        if False:
            for i in range(10):
                print('nop')
        always99 = functools.partial(self.maximum, 99, 20)
        always20 = functools.partial(self.maximum, 9, 20)
        self.assertEqual(__, always99())
        self.assertEqual(__, always20())

    class doubleit:

        def __init__(self, fn):
            if False:
                print('Hello World!')
            self.fn = fn

        def __call__(self, *args):
            if False:
                print('Hello World!')
            return self.fn(*args) + ', ' + self.fn(*args)

        def __get__(self, obj, cls=None):
            if False:
                print('Hello World!')
            if not obj:
                return self
            else:
                return functools.partial(self, obj)

    @doubleit
    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    @doubleit
    def parrot(self, text):
        if False:
            print('Hello World!')
        return text.upper()

    def test_decorator_with_no_arguments(self):
        if False:
            return 10
        self.assertEqual(__, self.foo())
        self.assertEqual(__, self.parrot('pieces of eight'))

    def sound_check(self):
        if False:
            return 10
        return 'Testing...'

    def test_what_a_decorator_is_doing_to_a_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.sound_check = self.doubleit(self.sound_check)
        self.assertEqual(__, self.sound_check())

    class documenter:

        def __init__(self, *args):
            if False:
                return 10
            self.fn_doc = args[0]

        def __call__(self, fn):
            if False:
                return 10

            def decorated_function(*args):
                if False:
                    print('Hello World!')
                return fn(*args)
            if fn.__doc__:
                decorated_function.__doc__ = fn.__doc__ + ': ' + self.fn_doc
            else:
                decorated_function.__doc__ = self.fn_doc
            return decorated_function

    @documenter('Increments a value by one. Kind of.')
    def count_badly(self, num):
        if False:
            i = 10
            return i + 15
        num += 1
        if num == 3:
            return 5
        else:
            return num

    @documenter('Does nothing')
    def idler(self, num):
        if False:
            i = 10
            return i + 15
        'Idler'
        pass

    def test_decorator_with_an_argument(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.count_badly(2))
        self.assertEqual(__, self.count_badly.__doc__)

    def test_documentor_which_already_has_a_docstring(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.idler.__doc__)

    @documenter('DOH!')
    @doubleit
    @doubleit
    def homer(self):
        if False:
            for i in range(10):
                print('nop')
        return "D'oh"

    def test_we_can_chain_decorators(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.homer())
        self.assertEqual(__, self.homer.__doc__)
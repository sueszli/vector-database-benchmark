from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['wrapt'])
def test_wrapt(selenium):
    if False:
        while True:
            i = 10
    import inspect
    import unittest
    import wrapt

    @wrapt.decorator
    def passthru_decorator(wrapped, instance, args, kwargs):
        if False:
            print('Hello World!')
        return wrapped(*args, **kwargs)

    def function1(arg):
        if False:
            while True:
                i = 10
        'documentation'
        return arg
    function1o = function1
    function1d = passthru_decorator(function1)
    assert function1d is not function1o

    class TestNamingFunction(unittest.TestCase):

        def test_object_name(self):
            if False:
                print('Hello World!')
            self.assertEqual(function1d.__name__, function1o.__name__)

        def test_object_qualname(self):
            if False:
                i = 10
                return i + 15
            try:
                __qualname__ = function1o.__qualname__
            except AttributeError:
                pass
            else:
                self.assertEqual(function1d.__qualname__, __qualname__)

        def test_module_name(self):
            if False:
                return 10
            self.assertEqual(function1d.__module__, __name__)

        def test_doc_string(self):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(function1d.__doc__, function1o.__doc__)

        def test_argspec(self):
            if False:
                return 10
            function1o_argspec = inspect.getargspec(function1o)
            function1d_argspec = inspect.getargspec(function1d)
            self.assertEqual(function1o_argspec, function1d_argspec)

        def test_isinstance(self):
            if False:
                return 10
            self.assertTrue(isinstance(function1d, type(function1o)))

    class TestCallingFunction(unittest.TestCase):

        def test_call_function(self):
            if False:
                while True:
                    i = 10
            _args = (1, 2)
            _kwargs = {'one': 1, 'two': 2}

            @wrapt.decorator
            def _decorator(wrapped, instance, args, kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertEqual(instance, None)
                self.assertEqual(args, _args)
                self.assertEqual(kwargs, _kwargs)
                return wrapped(*args, **kwargs)

            @_decorator
            def _function(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return (args, kwargs)
            result = _function(*_args, **_kwargs)
            self.assertEqual(result, (_args, _kwargs))
    with unittest.TestCase().assertRaisesRegex(SystemExit, 'False'):
        unittest.main()
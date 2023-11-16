import textwrap
import types
import unittest

def global_function():
    if False:
        print('Hello World!')

    def inner_function():
        if False:
            print('Hello World!')

        class LocalClass:
            pass
        global inner_global_function

        def inner_global_function():
            if False:
                return 10

            def inner_function2():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            return inner_function2
        return LocalClass
    return lambda : inner_function

class FuncAttrsTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10

        class F:

            def a(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        def b():
            if False:
                for i in range(10):
                    print('nop')
            return 3
        self.fi = F()
        self.F = F
        self.b = b

    def cannot_set_attr(self, obj, name, value, exceptions):
        if False:
            print('Hello World!')
        try:
            setattr(obj, name, value)
        except exceptions:
            pass
        else:
            self.fail("shouldn't be able to set %s to %r" % (name, value))
        try:
            delattr(obj, name)
        except exceptions:
            pass
        else:
            self.fail("shouldn't be able to del %s" % name)

class FunctionPropertiesTest(FuncAttrsTest):

    def test_module(self):
        if False:
            return 10
        self.assertEqual(self.b.__module__, __name__)

    def test_dir_includes_correct_attrs(self):
        if False:
            i = 10
            return i + 15
        self.b.known_attr = 7
        self.assertIn('known_attr', dir(self.b), 'set attributes not in dir listing of method')
        self.F.a.known_attr = 7
        self.assertIn('known_attr', dir(self.fi.a), 'set attribute on function implementations, should show up in next dir')

    def test_duplicate_function_equality(self):
        if False:
            while True:
                i = 10

        def duplicate():
            if False:
                for i in range(10):
                    print('nop')
            'my docstring'
            return 3
        self.assertNotEqual(self.b, duplicate)

    def test_copying___code__(self):
        if False:
            print('Hello World!')

        def test():
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(test(), None)
        test.__code__ = self.b.__code__
        self.assertEqual(test(), 3)

    def test___globals__(self):
        if False:
            return 10
        self.assertIs(self.b.__globals__, globals())
        self.cannot_set_attr(self.b, '__globals__', 2, (AttributeError, TypeError))

    def test___builtins__(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(self.b.__builtins__, __builtins__)
        self.cannot_set_attr(self.b, '__builtins__', 2, (AttributeError, TypeError))

        def func(s):
            if False:
                for i in range(10):
                    print('nop')
            return len(s)
        ns = {}
        func2 = type(func)(func.__code__, ns)
        self.assertIs(func2.__globals__, ns)
        self.assertIs(func2.__builtins__, __builtins__)
        self.assertEqual(func2('abc'), 3)
        self.assertEqual(ns, {})
        code = textwrap.dedent('\n            def func3(s): pass\n            func4 = type(func3)(func3.__code__, {})\n        ')
        safe_builtins = {'None': None}
        ns = {'type': type, '__builtins__': safe_builtins}
        exec(code, ns)
        self.assertIs(ns['func3'].__builtins__, safe_builtins)
        self.assertIs(ns['func4'].__builtins__, safe_builtins)
        self.assertIs(ns['func3'].__globals__['__builtins__'], safe_builtins)
        self.assertNotIn('__builtins__', ns['func4'].__globals__)

    def test___closure__(self):
        if False:
            for i in range(10):
                print('nop')
        a = 12

        def f():
            if False:
                return 10
            print(a)
        c = f.__closure__
        self.assertIsInstance(c, tuple)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0].__class__.__name__, 'cell')
        self.cannot_set_attr(f, '__closure__', c, AttributeError)

    def test_cell_new(self):
        if False:
            while True:
                i = 10
        cell_obj = types.CellType(1)
        self.assertEqual(cell_obj.cell_contents, 1)
        cell_obj = types.CellType()
        msg = "shouldn't be able to read an empty cell"
        with self.assertRaises(ValueError, msg=msg):
            cell_obj.cell_contents

    def test_empty_cell(self):
        if False:
            return 10

        def f():
            if False:
                return 10
            print(a)
        try:
            f.__closure__[0].cell_contents
        except ValueError:
            pass
        else:
            self.fail("shouldn't be able to read an empty cell")
        a = 12

    def test_set_cell(self):
        if False:
            i = 10
            return i + 15
        a = 12

        def f():
            if False:
                print('Hello World!')
            return a
        c = f.__closure__
        c[0].cell_contents = 9
        self.assertEqual(c[0].cell_contents, 9)
        self.assertEqual(f(), 9)
        self.assertEqual(a, 9)
        del c[0].cell_contents
        try:
            c[0].cell_contents
        except ValueError:
            pass
        else:
            self.fail("shouldn't be able to read an empty cell")
        with self.assertRaises(NameError):
            f()
        with self.assertRaises(UnboundLocalError):
            print(a)

    def test___name__(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.b.__name__, 'b')
        self.b.__name__ = 'c'
        self.assertEqual(self.b.__name__, 'c')
        self.b.__name__ = 'd'
        self.assertEqual(self.b.__name__, 'd')
        self.cannot_set_attr(self.b, '__name__', 7, TypeError)
        s = 'def f(): pass\nf.__name__'
        exec(s, {'__builtins__': {}})
        self.assertEqual(self.fi.a.__name__, 'a')
        self.cannot_set_attr(self.fi.a, '__name__', 'a', AttributeError)

    def test___qualname__(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.b.__qualname__, 'FuncAttrsTest.setUp.<locals>.b')
        self.assertEqual(FuncAttrsTest.setUp.__qualname__, 'FuncAttrsTest.setUp')
        self.assertEqual(global_function.__qualname__, 'global_function')
        self.assertEqual(global_function().__qualname__, 'global_function.<locals>.<lambda>')
        self.assertEqual(global_function()().__qualname__, 'global_function.<locals>.inner_function')
        self.assertEqual(global_function()()().__qualname__, 'global_function.<locals>.inner_function.<locals>.LocalClass')
        self.assertEqual(inner_global_function.__qualname__, 'inner_global_function')
        self.assertEqual(inner_global_function().__qualname__, 'inner_global_function.<locals>.inner_function2')
        self.b.__qualname__ = 'c'
        self.assertEqual(self.b.__qualname__, 'c')
        self.b.__qualname__ = 'd'
        self.assertEqual(self.b.__qualname__, 'd')
        self.cannot_set_attr(self.b, '__qualname__', 7, TypeError)

    def test___code__(self):
        if False:
            for i in range(10):
                print('nop')
        (num_one, num_two) = (7, 8)

        def a():
            if False:
                print('Hello World!')
            pass

        def b():
            if False:
                while True:
                    i = 10
            return 12

        def c():
            if False:
                i = 10
                return i + 15
            return num_one

        def d():
            if False:
                for i in range(10):
                    print('nop')
            return num_two

        def e():
            if False:
                while True:
                    i = 10
            return (num_one, num_two)
        for func in [a, b, c, d, e]:
            self.assertEqual(type(func.__code__), types.CodeType)
        self.assertEqual(c(), 7)
        self.assertEqual(d(), 8)
        d.__code__ = c.__code__
        self.assertEqual(c.__code__, d.__code__)
        self.assertEqual(c(), 7)
        try:
            b.__code__ = c.__code__
        except ValueError:
            pass
        else:
            self.fail('__code__ with different numbers of free vars should not be possible')
        try:
            e.__code__ = d.__code__
        except ValueError:
            pass
        else:
            self.fail('__code__ with different numbers of free vars should not be possible')

    def test_blank_func_defaults(self):
        if False:
            return 10
        self.assertEqual(self.b.__defaults__, None)
        del self.b.__defaults__
        self.assertEqual(self.b.__defaults__, None)

    def test_func_default_args(self):
        if False:
            print('Hello World!')

        def first_func(a, b):
            if False:
                i = 10
                return i + 15
            return a + b

        def second_func(a=1, b=2):
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        self.assertEqual(first_func.__defaults__, None)
        self.assertEqual(second_func.__defaults__, (1, 2))
        first_func.__defaults__ = (1, 2)
        self.assertEqual(first_func.__defaults__, (1, 2))
        self.assertEqual(first_func(), 3)
        self.assertEqual(first_func(3), 5)
        self.assertEqual(first_func(3, 5), 8)
        del second_func.__defaults__
        self.assertEqual(second_func.__defaults__, None)
        try:
            second_func()
        except TypeError:
            pass
        else:
            self.fail('__defaults__ does not update; deleting it does not remove requirement')

class InstancemethodAttrTest(FuncAttrsTest):

    def test___class__(self):
        if False:
            return 10
        self.assertEqual(self.fi.a.__self__.__class__, self.F)
        self.cannot_set_attr(self.fi.a, '__class__', self.F, TypeError)

    def test___func__(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.fi.a.__func__, self.F.a)
        self.cannot_set_attr(self.fi.a, '__func__', self.F.a, AttributeError)

    def test___self__(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.fi.a.__self__, self.fi)
        self.cannot_set_attr(self.fi.a, '__self__', self.fi, AttributeError)

    def test___func___non_method(self):
        if False:
            while True:
                i = 10
        self.fi.id = types.MethodType(id, self.fi)
        self.assertEqual(self.fi.id(), id(self.fi))
        try:
            self.fi.id.unknown_attr
        except AttributeError:
            pass
        else:
            self.fail('using unknown attributes should raise AttributeError')
        self.cannot_set_attr(self.fi.id, 'unknown_attr', 2, AttributeError)

class ArbitraryFunctionAttrTest(FuncAttrsTest):

    def test_set_attr(self):
        if False:
            return 10
        self.b.known_attr = 7
        self.assertEqual(self.b.known_attr, 7)
        try:
            self.fi.a.known_attr = 7
        except AttributeError:
            pass
        else:
            self.fail('setting attributes on methods should raise error')

    def test_delete_unknown_attr(self):
        if False:
            print('Hello World!')
        try:
            del self.b.unknown_attr
        except AttributeError:
            pass
        else:
            self.fail('deleting unknown attribute should raise TypeError')

    def test_unset_attr(self):
        if False:
            while True:
                i = 10
        for func in [self.b, self.fi.a]:
            try:
                func.non_existent_attr
            except AttributeError:
                pass
            else:
                self.fail('using unknown attributes should raise AttributeError')

class FunctionDictsTest(FuncAttrsTest):

    def test_setting_dict_to_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        self.cannot_set_attr(self.b, '__dict__', None, TypeError)
        from collections import UserDict
        d = UserDict({'known_attr': 7})
        self.cannot_set_attr(self.fi.a.__func__, '__dict__', d, TypeError)

    def test_setting_dict_to_valid(self):
        if False:
            return 10
        d = {'known_attr': 7}
        self.b.__dict__ = d
        self.assertIs(d, self.b.__dict__)
        self.F.a.__dict__ = d
        self.assertIs(d, self.fi.a.__func__.__dict__)
        self.assertIs(d, self.fi.a.__dict__)
        self.assertEqual(self.b.known_attr, 7)
        self.assertEqual(self.b.__dict__['known_attr'], 7)
        self.assertEqual(self.fi.a.__func__.known_attr, 7)
        self.assertEqual(self.fi.a.known_attr, 7)

    def test_delete___dict__(self):
        if False:
            return 10
        try:
            del self.b.__dict__
        except TypeError:
            pass
        else:
            self.fail('deleting function dictionary should raise TypeError')

    def test_unassigned_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.b.__dict__, {})

    def test_func_as_dict_key(self):
        if False:
            return 10
        value = 'Some string'
        d = {}
        d[self.b] = value
        self.assertEqual(d[self.b], value)

class FunctionDocstringTest(FuncAttrsTest):

    def test_set_docstring_attr(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.b.__doc__, None)
        docstr = 'A test method that does nothing'
        self.b.__doc__ = docstr
        self.F.a.__doc__ = docstr
        self.assertEqual(self.b.__doc__, docstr)
        self.assertEqual(self.fi.a.__doc__, docstr)
        self.cannot_set_attr(self.fi.a, '__doc__', docstr, AttributeError)

    def test_delete_docstring(self):
        if False:
            while True:
                i = 10
        self.b.__doc__ = 'The docstring'
        del self.b.__doc__
        self.assertEqual(self.b.__doc__, None)

def cell(value):
    if False:
        for i in range(10):
            print('nop')
    'Create a cell containing the given value.'

    def f():
        if False:
            for i in range(10):
                print('nop')
        print(a)
    a = value
    return f.__closure__[0]

def empty_cell(empty=True):
    if False:
        i = 10
        return i + 15
    'Create an empty cell.'

    def f():
        if False:
            return 10
        print(a)
    if not empty:
        a = 1729
    return f.__closure__[0]

class CellTest(unittest.TestCase):

    def test_comparison(self):
        if False:
            while True:
                i = 10
        self.assertTrue(cell(2) < cell(3))
        self.assertTrue(empty_cell() < cell('saturday'))
        self.assertTrue(empty_cell() == empty_cell())
        self.assertTrue(cell(-36) == cell(-36.0))
        self.assertTrue(cell(True) > empty_cell())

class StaticMethodAttrsTest(unittest.TestCase):

    def test_func_attribute(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            pass
        c = classmethod(f)
        self.assertTrue(c.__func__ is f)
        s = staticmethod(f)
        self.assertTrue(s.__func__ is f)

class BuiltinFunctionPropertiesTest(unittest.TestCase):

    def test_builtin__qualname__(self):
        if False:
            while True:
                i = 10
        import time
        self.assertEqual(len.__qualname__, 'len')
        self.assertEqual(time.time.__qualname__, 'time')
        self.assertEqual(dict.fromkeys.__qualname__, 'dict.fromkeys')
        self.assertEqual(float.__getformat__.__qualname__, 'float.__getformat__')
        self.assertEqual(str.maketrans.__qualname__, 'str.maketrans')
        self.assertEqual(bytes.maketrans.__qualname__, 'bytes.maketrans')
        self.assertEqual([1, 2, 3].append.__qualname__, 'list.append')
        self.assertEqual({'foo': 'bar'}.pop.__qualname__, 'dict.pop')
if __name__ == '__main__':
    unittest.main()
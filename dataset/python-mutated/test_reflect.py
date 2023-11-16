"""
Test cases for the L{twisted.python.reflect} module.
"""
import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import accumulateMethods, addMethodNamesToDict, fullyQualifiedName, prefixedMethodNames, prefixedMethods
from twisted.trial.unittest import SynchronousTestCase as TestCase

class Base:
    """
    A no-op class which can be used to verify the behavior of
    method-discovering APIs.
    """

    def method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A no-op method which can be discovered.\n        '

class Sub(Base):
    """
    A subclass of a class with a method which can be discovered.
    """

class Separate:
    """
    A no-op class with methods with differing prefixes.
    """

    def good_method(self):
        if False:
            i = 10
            return i + 15
        '\n        A no-op method which a matching prefix to be discovered.\n        '

    def bad_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A no-op method with a mismatched prefix to not be discovered.\n        '

class AccumulateMethodsTests(TestCase):
    """
    Tests for L{accumulateMethods} which finds methods on a class hierarchy and
    adds them to a dictionary.
    """

    def test_ownClass(self):
        if False:
            while True:
                i = 10
        '\n        If x is and instance of Base and Base defines a method named method,\n        L{accumulateMethods} adds an item to the given dictionary with\n        C{"method"} as the key and a bound method object for Base.method value.\n        '
        x = Base()
        output = {}
        accumulateMethods(x, output)
        self.assertEqual({'method': x.method}, output)

    def test_baseClass(self):
        if False:
            return 10
        '\n        If x is an instance of Sub and Sub is a subclass of Base and Base\n        defines a method named method, L{accumulateMethods} adds an item to the\n        given dictionary with C{"method"} as the key and a bound method object\n        for Base.method as the value.\n        '
        x = Sub()
        output = {}
        accumulateMethods(x, output)
        self.assertEqual({'method': x.method}, output)

    def test_prefix(self):
        if False:
            return 10
        '\n        If a prefix is given, L{accumulateMethods} limits its results to\n        methods beginning with that prefix.  Keys in the resulting dictionary\n        also have the prefix removed from them.\n        '
        x = Separate()
        output = {}
        accumulateMethods(x, output, 'good_')
        self.assertEqual({'method': x.good_method}, output)

class PrefixedMethodsTests(TestCase):
    """
    Tests for L{prefixedMethods} which finds methods on a class hierarchy and
    adds them to a dictionary.
    """

    def test_onlyObject(self):
        if False:
            i = 10
            return i + 15
        '\n        L{prefixedMethods} returns a list of the methods discovered on an\n        object.\n        '
        x = Base()
        output = prefixedMethods(x)
        self.assertEqual([x.method], output)

    def test_prefix(self):
        if False:
            i = 10
            return i + 15
        '\n        If a prefix is given, L{prefixedMethods} returns only methods named\n        with that prefix.\n        '
        x = Separate()
        output = prefixedMethods(x, 'good_')
        self.assertEqual([x.good_method], output)

class PrefixedMethodNamesTests(TestCase):
    """
    Tests for L{prefixedMethodNames}.
    """

    def test_method(self):
        if False:
            i = 10
            return i + 15
        '\n        L{prefixedMethodNames} returns a list including methods with the given\n        prefix defined on the class passed to it.\n        '
        self.assertEqual(['method'], prefixedMethodNames(Separate, 'good_'))

    def test_inheritedMethod(self):
        if False:
            return 10
        '\n        L{prefixedMethodNames} returns a list included methods with the given\n        prefix defined on base classes of the class passed to it.\n        '

        class Child(Separate):
            pass
        self.assertEqual(['method'], prefixedMethodNames(Child, 'good_'))

class AddMethodNamesToDictTests(TestCase):
    """
    Tests for L{addMethodNamesToDict}.
    """

    def test_baseClass(self):
        if False:
            return 10
        '\n        If C{baseClass} is passed to L{addMethodNamesToDict}, only methods which\n        are a subclass of C{baseClass} are added to the result dictionary.\n        '

        class Alternate:
            pass

        class Child(Separate, Alternate):

            def good_alternate(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        result = {}
        addMethodNamesToDict(Child, result, 'good_', Alternate)
        self.assertEqual({'alternate': 1}, result)

class Summer:
    """
    A class we look up as part of the LookupsTests.
    """

    def reallySet(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do something.\n        '

class LookupsTests(TestCase):
    """
    Tests for L{namedClass}, L{namedModule}, and L{namedAny}.
    """

    def test_namedClassLookup(self):
        if False:
            i = 10
            return i + 15
        '\n        L{namedClass} should return the class object for the name it is passed.\n        '
        self.assertIs(reflect.namedClass('twisted.test.test_reflect.Summer'), Summer)

    def test_namedModuleLookup(self):
        if False:
            i = 10
            return i + 15
        '\n        L{namedModule} should return the module object for the name it is\n        passed.\n        '
        from twisted.python import monkey
        self.assertIs(reflect.namedModule('twisted.python.monkey'), monkey)

    def test_namedAnyPackageLookup(self):
        if False:
            print('Hello World!')
        '\n        L{namedAny} should return the package object for the name it is passed.\n        '
        import twisted.python
        self.assertIs(reflect.namedAny('twisted.python'), twisted.python)

    def test_namedAnyModuleLookup(self):
        if False:
            i = 10
            return i + 15
        '\n        L{namedAny} should return the module object for the name it is passed.\n        '
        from twisted.python import monkey
        self.assertIs(reflect.namedAny('twisted.python.monkey'), monkey)

    def test_namedAnyClassLookup(self):
        if False:
            while True:
                i = 10
        '\n        L{namedAny} should return the class object for the name it is passed.\n        '
        self.assertIs(reflect.namedAny('twisted.test.test_reflect.Summer'), Summer)

    def test_namedAnyAttributeLookup(self):
        if False:
            i = 10
            return i + 15
        '\n        L{namedAny} should return the object an attribute of a non-module,\n        non-package object is bound to for the name it is passed.\n        '
        self.assertEqual(reflect.namedAny('twisted.test.test_reflect.Summer.reallySet'), Summer.reallySet)

    def test_namedAnySecondAttributeLookup(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{namedAny} should return the object an attribute of an object which\n        itself was an attribute of a non-module, non-package object is bound to\n        for the name it is passed.\n        '
        self.assertIs(reflect.namedAny('twisted.test.test_reflect.Summer.reallySet.__doc__'), Summer.reallySet.__doc__)

    def test_importExceptions(self):
        if False:
            i = 10
            return i + 15
        '\n        Exceptions raised by modules which L{namedAny} causes to be imported\n        should pass through L{namedAny} to the caller.\n        '
        self.assertRaises(ZeroDivisionError, reflect.namedAny, 'twisted.test.reflect_helper_ZDE')
        self.assertRaises(ZeroDivisionError, reflect.namedAny, 'twisted.test.reflect_helper_ZDE')
        self.assertRaises(ValueError, reflect.namedAny, 'twisted.test.reflect_helper_VE')
        self.assertRaises(ImportError, reflect.namedAny, 'twisted.test.reflect_helper_IE')

    def test_attributeExceptions(self):
        if False:
            print('Hello World!')
        "\n        If segments on the end of a fully-qualified Python name represents\n        attributes which aren't actually present on the object represented by\n        the earlier segments, L{namedAny} should raise an L{AttributeError}.\n        "
        self.assertRaises(AttributeError, reflect.namedAny, 'twisted.nosuchmoduleintheworld')
        self.assertRaises(AttributeError, reflect.namedAny, 'twisted.nosuch.modulein.theworld')
        self.assertRaises(AttributeError, reflect.namedAny, 'twisted.test.test_reflect.Summer.nosuchattribute')

    def test_invalidNames(self):
        if False:
            print('Hello World!')
        "\n        Passing a name which isn't a fully-qualified Python name to L{namedAny}\n        should result in one of the following exceptions:\n         - L{InvalidName}: the name is not a dot-separated list of Python\n           objects\n         - L{ObjectNotFound}: the object doesn't exist\n         - L{ModuleNotFound}: the object doesn't exist and there is only one\n           component in the name\n        "
        err = self.assertRaises(reflect.ModuleNotFound, reflect.namedAny, 'nosuchmoduleintheworld')
        self.assertEqual(str(err), "No module named 'nosuchmoduleintheworld'")
        err = self.assertRaises(reflect.ObjectNotFound, reflect.namedAny, '@#$@(#.!@(#!@#')
        self.assertEqual(str(err), "'@#$@(#.!@(#!@#' does not name an object")
        err = self.assertRaises(reflect.ObjectNotFound, reflect.namedAny, 'tcelfer.nohtyp.detsiwt')
        self.assertEqual(str(err), "'tcelfer.nohtyp.detsiwt' does not name an object")
        err = self.assertRaises(reflect.InvalidName, reflect.namedAny, '')
        self.assertEqual(str(err), 'Empty module name')
        for invalidName in ['.twisted', 'twisted.', 'twisted..python']:
            err = self.assertRaises(reflect.InvalidName, reflect.namedAny, invalidName)
            self.assertEqual(str(err), "name must be a string giving a '.'-separated list of Python identifiers, not %r" % (invalidName,))

    def test_requireModuleImportError(self):
        if False:
            return 10
        '\n        When module import fails with ImportError it returns the specified\n        default value.\n        '
        for name in ['nosuchmtopodule', 'no.such.module']:
            default = object()
            result = reflect.requireModule(name, default=default)
            self.assertIs(result, default)

    def test_requireModuleDefaultNone(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When module import fails it returns L{None} by default.\n        '
        result = reflect.requireModule('no.such.module')
        self.assertIsNone(result)

    def test_requireModuleRequestedImport(self):
        if False:
            return 10
        '\n        When module import succeed it returns the module and not the default\n        value.\n        '
        from twisted.python import monkey
        default = object()
        self.assertIs(reflect.requireModule('twisted.python.monkey', default=default), monkey)

class Breakable:
    breakRepr = False
    breakStr = False

    def __str__(self) -> str:
        if False:
            return 10
        if self.breakStr:
            raise RuntimeError('str!')
        else:
            return '<Breakable>'

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.breakRepr:
            raise RuntimeError('repr!')
        else:
            return 'Breakable()'

class BrokenType(Breakable, type):
    breakName = False

    @property
    def __name__(self):
        if False:
            i = 10
            return i + 15
        if self.breakName:
            raise RuntimeError('no name')
        return 'BrokenType'
BTBase = BrokenType('BTBase', (Breakable,), {'breakRepr': True, 'breakStr': True})

class NoClassAttr(Breakable):
    __class__ = property(lambda x: x.not_class)

class SafeReprTests(TestCase):
    """
    Tests for L{reflect.safe_repr} function.
    """

    def test_workingRepr(self):
        if False:
            return 10
        '\n        L{reflect.safe_repr} produces the same output as C{repr} on a working\n        object.\n        '
        xs = ([1, 2, 3], b'a')
        self.assertEqual(list(map(reflect.safe_repr, xs)), list(map(repr, xs)))

    def test_brokenRepr(self):
        if False:
            while True:
                i = 10
        '\n        L{reflect.safe_repr} returns a string with class name, address, and\n        traceback when the repr call failed.\n        '
        b = Breakable()
        b.breakRepr = True
        bRepr = reflect.safe_repr(b)
        self.assertIn('Breakable instance at 0x', bRepr)
        self.assertIn(os.path.splitext(__file__)[0], bRepr)
        self.assertIn('RuntimeError: repr!', bRepr)

    def test_brokenStr(self):
        if False:
            while True:
                i = 10
        "\n        L{reflect.safe_repr} isn't affected by a broken C{__str__} method.\n        "
        b = Breakable()
        b.breakStr = True
        self.assertEqual(reflect.safe_repr(b), repr(b))

    def test_brokenClassRepr(self):
        if False:
            while True:
                i = 10

        class X(BTBase):
            breakRepr = True
        reflect.safe_repr(X)
        reflect.safe_repr(X())

    def test_brokenReprIncludesID(self):
        if False:
            i = 10
            return i + 15
        '\n        C{id} is used to print the ID of the object in case of an error.\n\n        L{safe_repr} includes a traceback after a newline, so we only check\n        against the first line of the repr.\n        '

        class X(BTBase):
            breakRepr = True
        xRepr = reflect.safe_repr(X)
        xReprExpected = f'<BrokenType instance at 0x{id(X):x} with repr error:'
        self.assertEqual(xReprExpected, xRepr.split('\n')[0])

    def test_brokenClassStr(self):
        if False:
            i = 10
            return i + 15

        class X(BTBase):
            breakStr = True
        reflect.safe_repr(X)
        reflect.safe_repr(X())

    def test_brokenClassAttribute(self):
        if False:
            while True:
                i = 10
        '\n        If an object raises an exception when accessing its C{__class__}\n        attribute, L{reflect.safe_repr} uses C{type} to retrieve the class\n        object.\n        '
        b = NoClassAttr()
        b.breakRepr = True
        bRepr = reflect.safe_repr(b)
        self.assertIn('NoClassAttr instance at 0x', bRepr)
        self.assertIn(os.path.splitext(__file__)[0], bRepr)
        self.assertIn('RuntimeError: repr!', bRepr)

    def test_brokenClassNameAttribute(self):
        if False:
            while True:
                i = 10
        "\n        If a class raises an exception when accessing its C{__name__} attribute\n        B{and} when calling its C{__str__} implementation, L{reflect.safe_repr}\n        returns 'BROKEN CLASS' instead of the class name.\n        "

        class X(BTBase):
            breakName = True
        xRepr = reflect.safe_repr(X())
        self.assertIn('<BROKEN CLASS AT 0x', xRepr)
        self.assertIn(os.path.splitext(__file__)[0], xRepr)
        self.assertIn('RuntimeError: repr!', xRepr)

class SafeStrTests(TestCase):
    """
    Tests for L{reflect.safe_str} function.
    """

    def test_workingStr(self):
        if False:
            print('Hello World!')
        x = [1, 2, 3]
        self.assertEqual(reflect.safe_str(x), str(x))

    def test_brokenStr(self):
        if False:
            for i in range(10):
                print('nop')
        b = Breakable()
        b.breakStr = True
        reflect.safe_str(b)

    def test_workingAscii(self):
        if False:
            return 10
        '\n        L{safe_str} for C{str} with ascii-only data should return the\n        value unchanged.\n        '
        x = 'a'
        self.assertEqual(reflect.safe_str(x), 'a')

    def test_workingUtf8_3(self):
        if False:
            return 10
        '\n        L{safe_str} for C{bytes} with utf-8 encoded data should return\n        the value decoded into C{str}.\n        '
        x = b't\xc3\xbcst'
        self.assertEqual(reflect.safe_str(x), x.decode('utf-8'))

    def test_brokenUtf8(self):
        if False:
            return 10
        '\n        Use str() for non-utf8 bytes: "b\'non-utf8\'"\n        '
        x = b'\xff'
        xStr = reflect.safe_str(x)
        self.assertEqual(xStr, str(x))

    def test_brokenRepr(self):
        if False:
            while True:
                i = 10
        b = Breakable()
        b.breakRepr = True
        reflect.safe_str(b)

    def test_brokenClassStr(self):
        if False:
            return 10

        class X(BTBase):
            breakStr = True
        reflect.safe_str(X)
        reflect.safe_str(X())

    def test_brokenClassRepr(self):
        if False:
            for i in range(10):
                print('nop')

        class X(BTBase):
            breakRepr = True
        reflect.safe_str(X)
        reflect.safe_str(X())

    def test_brokenClassAttribute(self):
        if False:
            i = 10
            return i + 15
        '\n        If an object raises an exception when accessing its C{__class__}\n        attribute, L{reflect.safe_str} uses C{type} to retrieve the class\n        object.\n        '
        b = NoClassAttr()
        b.breakStr = True
        bStr = reflect.safe_str(b)
        self.assertIn('NoClassAttr instance at 0x', bStr)
        self.assertIn(os.path.splitext(__file__)[0], bStr)
        self.assertIn('RuntimeError: str!', bStr)

    def test_brokenClassNameAttribute(self):
        if False:
            print('Hello World!')
        "\n        If a class raises an exception when accessing its C{__name__} attribute\n        B{and} when calling its C{__str__} implementation, L{reflect.safe_str}\n        returns 'BROKEN CLASS' instead of the class name.\n        "

        class X(BTBase):
            breakName = True
        xStr = reflect.safe_str(X())
        self.assertIn('<BROKEN CLASS AT 0x', xStr)
        self.assertIn(os.path.splitext(__file__)[0], xStr)
        self.assertIn('RuntimeError: str!', xStr)

class FilenameToModuleTests(TestCase):
    """
    Test L{filenameToModuleName} detection.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.path = os.path.join(self.mktemp(), 'fakepackage', 'test')
        os.makedirs(self.path)
        with open(os.path.join(self.path, '__init__.py'), 'w') as f:
            f.write('')
        with open(os.path.join(os.path.dirname(self.path), '__init__.py'), 'w') as f:
            f.write('')

    def test_directory(self):
        if False:
            return 10
        '\n        L{filenameToModuleName} returns the correct module (a package) given a\n        directory.\n        '
        module = reflect.filenameToModuleName(self.path)
        self.assertEqual(module, 'fakepackage.test')
        module = reflect.filenameToModuleName(self.path + os.path.sep)
        self.assertEqual(module, 'fakepackage.test')

    def test_file(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{filenameToModuleName} returns the correct module given the path to\n        its file.\n        '
        module = reflect.filenameToModuleName(os.path.join(self.path, 'test_reflect.py'))
        self.assertEqual(module, 'fakepackage.test.test_reflect')

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{filenameToModuleName} returns the correct module given a C{bytes}\n        path to its file.\n        '
        module = reflect.filenameToModuleName(os.path.join(self.path.encode('utf-8'), b'test_reflect.py'))
        self.assertEqual(module, 'fakepackage.test.test_reflect')

class FullyQualifiedNameTests(TestCase):
    """
    Test for L{fullyQualifiedName}.
    """

    def _checkFullyQualifiedName(self, obj, expected):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper to check that fully qualified name of C{obj} results to\n        C{expected}.\n        '
        self.assertEqual(fullyQualifiedName(obj), expected)

    def test_package(self):
        if False:
            print('Hello World!')
        '\n        L{fullyQualifiedName} returns the full name of a package and a\n        subpackage.\n        '
        import twisted
        self._checkFullyQualifiedName(twisted, 'twisted')
        import twisted.python
        self._checkFullyQualifiedName(twisted.python, 'twisted.python')

    def test_module(self):
        if False:
            while True:
                i = 10
        '\n        L{fullyQualifiedName} returns the name of a module inside a package.\n        '
        import twisted.python.compat
        self._checkFullyQualifiedName(twisted.python.compat, 'twisted.python.compat')

    def test_class(self):
        if False:
            while True:
                i = 10
        '\n        L{fullyQualifiedName} returns the name of a class and its module.\n        '
        self._checkFullyQualifiedName(FullyQualifiedNameTests, f'{__name__}.FullyQualifiedNameTests')

    def test_function(self):
        if False:
            return 10
        '\n        L{fullyQualifiedName} returns the name of a function inside its module.\n        '
        self._checkFullyQualifiedName(fullyQualifiedName, 'twisted.python.reflect.fullyQualifiedName')

    def test_boundMethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{fullyQualifiedName} returns the name of a bound method inside its\n        class and its module.\n        '
        self._checkFullyQualifiedName(self.test_boundMethod, f'{__name__}.{self.__class__.__name__}.test_boundMethod')

    def test_unboundMethod(self):
        if False:
            i = 10
            return i + 15
        '\n        L{fullyQualifiedName} returns the name of an unbound method inside its\n        class and its module.\n        '
        self._checkFullyQualifiedName(self.__class__.test_unboundMethod, f'{__name__}.{self.__class__.__name__}.test_unboundMethod')

class ObjectGrepTests(TestCase):

    def test_dictionary(self):
        if False:
            i = 10
            return i + 15
        '\n        Test references search through a dictionary, as a key or as a value.\n        '
        o = object()
        d1 = {None: o}
        d2 = {o: None}
        self.assertIn('[None]', reflect.objgrep(d1, o, reflect.isSame))
        self.assertIn('{None}', reflect.objgrep(d2, o, reflect.isSame))

    def test_list(self):
        if False:
            print('Hello World!')
        '\n        Test references search through a list.\n        '
        o = object()
        L = [None, o]
        self.assertIn('[1]', reflect.objgrep(L, o, reflect.isSame))

    def test_tuple(self):
        if False:
            while True:
                i = 10
        '\n        Test references search through a tuple.\n        '
        o = object()
        T = (o, None)
        self.assertIn('[0]', reflect.objgrep(T, o, reflect.isSame))

    def test_instance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test references search through an object attribute.\n        '

        class Dummy:
            pass
        o = object()
        d = Dummy()
        d.o = o
        self.assertIn('.o', reflect.objgrep(d, o, reflect.isSame))

    def test_weakref(self):
        if False:
            i = 10
            return i + 15
        '\n        Test references search through a weakref object.\n        '

        class Dummy:
            pass
        o = Dummy()
        w1 = weakref.ref(o)
        self.assertIn('()', reflect.objgrep(w1, o, reflect.isSame))

    def test_boundMethod(self):
        if False:
            print('Hello World!')
        '\n        Test references search through method special attributes.\n        '

        class Dummy:

            def dummy(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        o = Dummy()
        m = o.dummy
        self.assertIn('.__self__', reflect.objgrep(m, m.__self__, reflect.isSame))
        self.assertIn('.__self__.__class__', reflect.objgrep(m, m.__self__.__class__, reflect.isSame))
        self.assertIn('.__func__', reflect.objgrep(m, m.__func__, reflect.isSame))

    def test_everything(self):
        if False:
            while True:
                i = 10
        '\n        Test references search using complex set of objects.\n        '

        class Dummy:

            def method(self):
                if False:
                    print('Hello World!')
                pass
        o = Dummy()
        D1 = {(): 'baz', None: 'Quux', o: 'Foosh'}
        L = [None, (), D1, 3]
        T = (L, {}, Dummy())
        D2 = {0: 'foo', 1: 'bar', 2: T}
        i = Dummy()
        i.attr = D2
        m = i.method
        w = weakref.ref(m)
        self.assertIn("().__self__.attr[2][0][2]{'Foosh'}", reflect.objgrep(w, o, reflect.isSame))

    def test_depthLimit(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the depth of references search.\n        '
        a = []
        b = [a]
        c = [a, b]
        d = [a, c]
        self.assertEqual(['[0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=1))
        self.assertEqual(['[0]', '[1][0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=2))
        self.assertEqual(['[0]', '[1][0]', '[1][1][0]'], reflect.objgrep(d, a, reflect.isSame, maxDepth=3))

    def test_deque(self):
        if False:
            print('Hello World!')
        '\n        Test references search through a deque object.\n        '
        o = object()
        D = deque()
        D.append(None)
        D.append(o)
        self.assertIn('[1]', reflect.objgrep(D, o, reflect.isSame))

class GetClassTests(TestCase):

    def test_new(self):
        if False:
            while True:
                i = 10

        class NewClass:
            pass
        new = NewClass()
        self.assertEqual(reflect.getClass(NewClass).__name__, 'type')
        self.assertEqual(reflect.getClass(new).__name__, 'NewClass')
import datetime
import unittest
from unittest import mock
from factory import base, declarations, errors, helpers
from . import utils

class OrderedDeclarationTestCase(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(NotImplementedError):
            utils.evaluate_declaration(declarations.OrderedDeclaration())

class DigTestCase(unittest.TestCase):

    class MyObj:

        def __init__(self, n):
            if False:
                return 10
            self.n = n

    def test_chaining(self):
        if False:
            i = 10
            return i + 15
        obj = self.MyObj(1)
        obj.a = self.MyObj(2)
        obj.a.b = self.MyObj(3)
        obj.a.b.c = self.MyObj(4)
        self.assertEqual(2, declarations.deepgetattr(obj, 'a').n)
        with self.assertRaises(AttributeError):
            declarations.deepgetattr(obj, 'b')
        self.assertEqual(2, declarations.deepgetattr(obj, 'a.n'))
        self.assertEqual(3, declarations.deepgetattr(obj, 'a.c', 3))
        with self.assertRaises(AttributeError):
            declarations.deepgetattr(obj, 'a.c.n')
        with self.assertRaises(AttributeError):
            declarations.deepgetattr(obj, 'a.d')
        self.assertEqual(3, declarations.deepgetattr(obj, 'a.b').n)
        self.assertEqual(3, declarations.deepgetattr(obj, 'a.b.n'))
        self.assertEqual(4, declarations.deepgetattr(obj, 'a.b.c').n)
        self.assertEqual(4, declarations.deepgetattr(obj, 'a.b.c.n'))
        self.assertEqual(42, declarations.deepgetattr(obj, 'a.b.c.n.x', 42))

class MaybeTestCase(unittest.TestCase):

    def test_init(self):
        if False:
            while True:
                i = 10
        declarations.Maybe('foo', 1, 2)
        with self.assertRaisesRegex(TypeError, 'Inconsistent phases'):
            declarations.Maybe('foo', declarations.LazyAttribute(None), declarations.PostGenerationDeclaration())

class SelfAttributeTestCase(unittest.TestCase):

    def test_standard(self):
        if False:
            return 10
        a = declarations.SelfAttribute('foo.bar.baz')
        self.assertEqual(0, a.depth)
        self.assertEqual('foo.bar.baz', a.attribute_name)
        self.assertEqual(declarations._UNSPECIFIED, a.default)

    def test_dot(self):
        if False:
            return 10
        a = declarations.SelfAttribute('.bar.baz')
        self.assertEqual(1, a.depth)
        self.assertEqual('bar.baz', a.attribute_name)
        self.assertEqual(declarations._UNSPECIFIED, a.default)

    def test_default(self):
        if False:
            i = 10
            return i + 15
        a = declarations.SelfAttribute('bar.baz', 42)
        self.assertEqual(0, a.depth)
        self.assertEqual('bar.baz', a.attribute_name)
        self.assertEqual(42, a.default)

    def test_parent(self):
        if False:
            for i in range(10):
                print('nop')
        a = declarations.SelfAttribute('..bar.baz')
        self.assertEqual(2, a.depth)
        self.assertEqual('bar.baz', a.attribute_name)
        self.assertEqual(declarations._UNSPECIFIED, a.default)

    def test_grandparent(self):
        if False:
            while True:
                i = 10
        a = declarations.SelfAttribute('...bar.baz')
        self.assertEqual(3, a.depth)
        self.assertEqual('bar.baz', a.attribute_name)
        self.assertEqual(declarations._UNSPECIFIED, a.default)

class IteratorTestCase(unittest.TestCase):

    def test_cycle(self):
        if False:
            print('Hello World!')
        it = declarations.Iterator([1, 2])
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=1))
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=2))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=3))

    def test_no_cycling(self):
        if False:
            print('Hello World!')
        it = declarations.Iterator([1, 2], cycle=False)
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=1))
        with self.assertRaises(StopIteration):
            utils.evaluate_declaration(it, force_sequence=2)

    def test_initial_reset(self):
        if False:
            print('Hello World!')
        it = declarations.Iterator([1, 2])
        it.reset()

    def test_reset_cycle(self):
        if False:
            i = 10
            return i + 15
        it = declarations.Iterator([1, 2])
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=1))
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=2))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=3))
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=4))
        it.reset()
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=5))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=6))

    def test_reset_no_cycling(self):
        if False:
            while True:
                i = 10
        it = declarations.Iterator([1, 2], cycle=False)
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=1))
        with self.assertRaises(StopIteration):
            utils.evaluate_declaration(it, force_sequence=2)
        it.reset()
        self.assertEqual(1, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=1))
        with self.assertRaises(StopIteration):
            utils.evaluate_declaration(it, force_sequence=2)

    def test_getter(self):
        if False:
            for i in range(10):
                print('nop')
        it = declarations.Iterator([(1, 2), (1, 3)], getter=lambda p: p[1])
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=0))
        self.assertEqual(3, utils.evaluate_declaration(it, force_sequence=1))
        self.assertEqual(2, utils.evaluate_declaration(it, force_sequence=2))
        self.assertEqual(3, utils.evaluate_declaration(it, force_sequence=3))

class TransformerTestCase(unittest.TestCase):

    def test_transform(self):
        if False:
            while True:
                i = 10
        t = declarations.Transformer('foo', transform=str.upper)
        self.assertEqual('FOO', utils.evaluate_declaration(t))

class PostGenerationDeclarationTestCase(unittest.TestCase):

    def test_post_generation(self):
        if False:
            return 10
        call_params = []

        def foo(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            call_params.append(args)
            call_params.append(kwargs)
        helpers.build(dict, foo=declarations.PostGeneration(foo), foo__bar=42, blah=42, blah__baz=1)
        self.assertEqual(2, len(call_params))
        self.assertEqual(3, len(call_params[0]))
        self.assertEqual({'bar': 42}, call_params[1])

    def test_decorator_simple(self):
        if False:
            return 10
        call_params = []

        @helpers.post_generation
        def foo(*args, **kwargs):
            if False:
                return 10
            call_params.append(args)
            call_params.append(kwargs)
        helpers.build(dict, foo=foo, foo__bar=42, blah=42, blah__baz=1)
        self.assertEqual(2, len(call_params))
        self.assertEqual(3, len(call_params[0]))
        self.assertEqual({'bar': 42}, call_params[1])

class FactoryWrapperTestCase(unittest.TestCase):

    def test_invalid_path(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            declarations._FactoryWrapper('UnqualifiedSymbol')
        with self.assertRaises(ValueError):
            declarations._FactoryWrapper(42)

    def test_class(self):
        if False:
            while True:
                i = 10
        w = declarations._FactoryWrapper(datetime.date)
        self.assertEqual(datetime.date, w.get())

    def test_path(self):
        if False:
            print('Hello World!')
        w = declarations._FactoryWrapper('datetime.date')
        self.assertEqual(datetime.date, w.get())

    def test_lazyness(self):
        if False:
            print('Hello World!')
        f = declarations._FactoryWrapper('factory.declarations.Sequence')
        self.assertEqual(None, f.factory)
        factory_class = f.get()
        self.assertEqual(declarations.Sequence, factory_class)

    def test_cache(self):
        if False:
            return 10
        'Ensure that _FactoryWrapper tries to import only once.'
        orig_date = datetime.date
        w = declarations._FactoryWrapper('datetime.date')
        self.assertEqual(None, w.factory)
        factory_class = w.get()
        self.assertEqual(orig_date, factory_class)
        try:
            datetime.date = None
            factory_class = w.get()
            self.assertEqual(orig_date, factory_class)
        finally:
            datetime.date = orig_date

class PostGenerationMethodCallTestCase(unittest.TestCase):

    def build(self, declaration, **params):
        if False:
            i = 10
            return i + 15
        f = helpers.make_factory(mock.MagicMock, post=declaration)
        return f(**params)

    def test_simplest_setup_and_call(self):
        if False:
            while True:
                i = 10
        obj = self.build(declarations.PostGenerationMethodCall('method'))
        obj.method.assert_called_once_with()

    def test_call_with_method_args(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.build(declarations.PostGenerationMethodCall('method', 'data'))
        obj.method.assert_called_once_with('data')

    def test_call_with_passed_extracted_string(self):
        if False:
            print('Hello World!')
        obj = self.build(declarations.PostGenerationMethodCall('method'), post='data')
        obj.method.assert_called_once_with('data')

    def test_call_with_passed_extracted_int(self):
        if False:
            i = 10
            return i + 15
        obj = self.build(declarations.PostGenerationMethodCall('method'), post=1)
        obj.method.assert_called_once_with(1)

    def test_call_with_passed_extracted_iterable(self):
        if False:
            while True:
                i = 10
        obj = self.build(declarations.PostGenerationMethodCall('method'), post=(1, 2, 3))
        obj.method.assert_called_once_with((1, 2, 3))

    def test_call_with_method_kwargs(self):
        if False:
            return 10
        obj = self.build(declarations.PostGenerationMethodCall('method', data='data'))
        obj.method.assert_called_once_with(data='data')

    def test_call_with_passed_kwargs(self):
        if False:
            print('Hello World!')
        obj = self.build(declarations.PostGenerationMethodCall('method'), post__data='other')
        obj.method.assert_called_once_with(data='other')

    def test_multi_call_with_multi_method_args(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(errors.InvalidDeclarationError):
            self.build(declarations.PostGenerationMethodCall('method', 'arg1', 'arg2'))

class PostGenerationOrdering(unittest.TestCase):

    def test_post_generation_declaration_order(self):
        if False:
            for i in range(10):
                print('nop')
        postgen_results = []

        class Related(base.Factory):

            class Meta:
                model = mock.MagicMock()

        class Ordered(base.Factory):

            class Meta:
                model = mock.MagicMock()
            a = declarations.RelatedFactory(Related)
            z = declarations.RelatedFactory(Related)

            @helpers.post_generation
            def a1(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                postgen_results.append('a1')

            @helpers.post_generation
            def zz(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                postgen_results.append('zz')

            @helpers.post_generation
            def aa(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                postgen_results.append('aa')
        postgen_names = Ordered._meta.post_declarations.sorted()
        self.assertEqual(postgen_names, ['a', 'z', 'a1', 'zz', 'aa'])
        Ordered()
        self.assertEqual(postgen_results, ['a1', 'zz', 'aa'])
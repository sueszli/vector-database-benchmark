"""Tests for type_inference module."""
from typing import Any, Callable, List
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import type_inference
from tensorflow.python.platform import test

class BasicTestResolver(type_inference.Resolver):
    """A very basic resolver for testing."""

    def res_name(self, ns, types_ns, name):
        if False:
            while True:
                i = 10
        str_name = str(name)
        if str_name == 'int':
            return ({int}, int)
        return ({type(ns[str_name])}, ns[str_name])

    def res_value(self, ns, value):
        if False:
            for i in range(10):
                print('nop')
        return {type(value)}

    def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
        if False:
            for i in range(10):
                print('nop')
        if type_anno is None:
            return None
        return {str(type_anno)}

class TestTranspiler(transpiler.GenericTranspiler):

    def __init__(self, resolver_type):
        if False:
            print('Hello World!')
        super().__init__()
        self.resolver = resolver_type()

    def get_transformed_name(self, _):
        if False:
            i = 10
            return i + 15
        return 'test_item'

    def transform_ast(self, node, ctx):
        if False:
            print('Hello World!')
        node = qual_names.resolve(node)
        node = activity.resolve(node, ctx)
        graphs = cfg.build(node)
        node = reaching_definitions.resolve(node, ctx, graphs)
        node = reaching_fndefs.resolve(node, ctx, graphs)
        node = type_inference.resolve(node, ctx, graphs, self.resolver)
        return node

class TypeInferenceAnalyzerTest(test.TestCase):

    def assertTypes(self, node, expected):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(expected, tuple):
            expected = (expected,)
        self.assertSetEqual(set(anno.getanno(node, anno.Static.TYPES)), set(expected))

    def assertClosureTypes(self, node, expected):
        if False:
            return 10
        actual = anno.getanno(node, anno.Static.CLOSURE_TYPES)
        actual = {str(k): v for (k, v) in actual.items()}
        for (k, v) in expected.items():
            self.assertIn(k, actual)
            self.assertEqual(actual[k], v)

    def test_no_inference_on_unknown_operand_types(self):
        if False:
            print('Hello World!')

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    i = 10
                    return i + 15
                return None

        def test_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a < b, a - b)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertFalse(anno.hasanno(fn_body[0].value.elts[0], anno.Static.TYPES))
        self.assertFalse(anno.hasanno(fn_body[0].value.elts[1], anno.Static.TYPES))

    def test_resolver_output_checked(self):
        if False:
            return 10

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    print('Hello World!')
                return 1

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            del a
            pass
        with self.assertRaisesRegex(ValueError, 'expected to return set'):
            TestTranspiler(Resolver).transform(test_fn, None)

    def test_argument(self):
        if False:
            print('Hello World!')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertFalse(f_is_local)
                if name == qual_names.QN('a'):
                    test_self.assertEqual(type_anno, qual_names.QN('int'))
                return {str(name) + '_type'}

        def test_fn(a: int, b):
            if False:
                i = 10
                return i + 15
            return (a, b)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.elts[0], 'a_type')
        self.assertTypes(fn_body[0].value.elts[1], 'b_type')

    def test_argument_of_local_function(self):
        if False:
            for i in range(10):
                print('nop')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                if f_name == 'test_fn':
                    test_self.assertFalse(f_is_local)
                    test_self.assertEqual(name, qual_names.QN('a'))
                    test_self.assertEqual(type_anno, qual_names.QN('int'))
                elif f_name == 'foo':
                    test_self.assertTrue(f_is_local)
                    if name == qual_names.QN('x'):
                        test_self.assertEqual(type_anno, qual_names.QN('float'))
                    elif name == qual_names.QN('y'):
                        test_self.assertIsNone(type_anno)
                    else:
                        test_self.fail('unexpected argument {} for {}'.format(name, f_name))
                else:
                    test_self.fail('unexpected function name {}'.format(f_name))
                return {str(name) + '_type'}

        def test_fn(a: int):
            if False:
                while True:
                    i = 10

            def foo(x: float, y):
                if False:
                    while True:
                        i = 10
                return (x, y)
            return foo(a, a)
        tr = TestTranspiler(Resolver)
        (node, _) = tr.transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[0].value, (('x_type', 'y_type'),))
        self.assertTypes(fn_body[0].body[0].value.elts[0], 'x_type')
        self.assertTypes(fn_body[0].body[0].value.elts[1], 'y_type')

    def test_assign_straightline(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a: int, c: float):
            if False:
                for i in range(10):
                    print('nop')
            b = a
            return (a, b, c)
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].targets[0], 'int')
        self.assertTypes(fn_body[0].value, 'int')
        self.assertTypes(fn_body[1].value.elts[0], 'int')
        self.assertTypes(fn_body[1].value.elts[1], 'int')
        self.assertTypes(fn_body[1].value.elts[2], 'float')

    def test_expr(self):
        if False:
            for i in range(10):
                print('nop')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_value(self, ns, value):
                if False:
                    return 10
                test_self.assertEqual(value, tc.a)
                return {str}

            def res_name(self, ns, types_ns, name):
                if False:
                    print('Hello World!')
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass}, tc)

            def res_call(self, ns, types_ns, node, f_type, args, keywords):
                if False:
                    return 10
                test_self.assertEqual(f_type, (str,))
                return ({int}, None)

        class TestClass:

            def a(self):
                if False:
                    i = 10
                    return i + 15
                pass
        tc = TestClass()

        def test_fn():
            if False:
                while True:
                    i = 10
            tc.a()
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertEqual(anno.getanno(fn_body[0].value.func, anno.Static.VALUE), tc.a)
        self.assertTypes(fn_body[0].value.func, str)
        self.assertTypes(fn_body[0].value, int)
        self.assertTypes(fn_body[0], int)

    def test_assign_overwriting(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a: int, b: float):
            if False:
                for i in range(10):
                    print('nop')
            c = a
            c = b
            return c
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].targets[0], 'int')
        self.assertTypes(fn_body[0].value, 'int')
        self.assertTypes(fn_body[1].targets[0], 'float')
        self.assertTypes(fn_body[1].value, 'float')

    def test_dynamic_attribute_of_static_value(self):
        if False:
            return 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_value(self, ns, value):
                if False:
                    print('Hello World!')
                test_self.assertEqual(value, tc.a)
                return {int}

            def res_name(self, ns, types_ns, name):
                if False:
                    while True:
                        i = 10
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass}, tc)

        class TestClass:

            def __init__(self):
                if False:
                    return 10
                self.a = 1
        tc = TestClass()

        def test_fn():
            if False:
                return 10
            return tc.a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.value, TestClass)
        self.assertTypes(fn_body[0].value, int)
        self.assertIs(anno.getanno(fn_body[0].value.value, anno.Static.VALUE), tc)
        self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), tc.a)

    def test_static_attribute_of_typed_value(self):
        if False:
            for i in range(10):
                print('nop')
        test_self = self

        class TestClass:
            a = 1
        tc = TestClass()

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass}, None)

            def res_value(self, ns, value):
                if False:
                    i = 10
                    return i + 15
                test_self.assertIs(value, tc.a)
                return {str}

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            return tc.a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.value, TestClass)
        self.assertTypes(fn_body[0].value, str)
        self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
        self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), 1)

    def test_static_attribute_of_ambiguous_type(self):
        if False:
            for i in range(10):
                print('nop')
        test_self = self

        class TestClass1:
            a = 1

        class TestClass2:
            a = 2
        tc = TestClass1()

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    i = 10
                    return i + 15
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass1, TestClass2}, None)

            def res_value(self, ns, value):
                if False:
                    print('Hello World!')
                test_self.assertIn(value, (1, 2))
                return {str}

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            return tc.a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.value, (TestClass1, TestClass2))
        self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
        self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
        self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

    def test_property_of_typed_value(self):
        if False:
            while True:
                i = 10
        test_self = self

        class TestClass:

            @property
            def a(self):
                if False:
                    while True:
                        i = 10
                return 1
        tc = TestClass()

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    i = 10
                    return i + 15
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass}, None)

            def res_value(self, ns, value):
                if False:
                    i = 10
                    return i + 15
                test_self.assertIs(value, TestClass.a)
                test_self.assertNotEqual(value, 1)
                return {property}

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            return tc.a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.value, TestClass)
        self.assertTypes(fn_body[0].value, property)
        self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
        self.assertEqual(anno.getanno(fn_body[0].value, anno.Static.VALUE), TestClass.a)

    def test_dynamic_attribute_of_typed_value(self):
        if False:
            for i in range(10):
                print('nop')
        test_self = self

        class TestClass:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.a = 1
        tc = TestClass()

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    print('Hello World!')
                test_self.assertEqual(name, qual_names.QN('tc'))
                return ({TestClass}, None)

        def test_fn():
            if False:
                return 10
            return tc.a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.value, TestClass)
        self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.TYPES))
        self.assertFalse(anno.hasanno(fn_body[0].value.value, anno.Static.VALUE))
        self.assertFalse(anno.hasanno(fn_body[0].value, anno.Static.VALUE))

    def test_external_value(self):
        if False:
            while True:
                i = 10
        a = 'foo'

        def test_fn():
            if False:
                return 10
            b = a
            return b
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].targets[0], str)
        self.assertTypes(fn_body[1].value, str)

    def test_external_function(self):
        if False:
            while True:
                i = 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    return 10
                test_self.assertEqual(name, qual_names.QN('g'))
                return ({str}, g)

            def res_call(self, ns, types_ns, node, f_type, args, keywords):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertEqual(f_type, (str,))
                test_self.assertEqual(anno.getanno(node.func, anno.Basic.QN), qual_names.QN('g'))
                return ({float}, None)

        def g() -> float:
            if False:
                i = 10
                return i + 15
            return 1.0

        def test_fn():
            if False:
                i = 10
                return i + 15
            a = g()
            return a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value.func, str)
        self.assertTypes(fn_body[0].targets[0], float)
        self.assertTypes(fn_body[1].value, float)

    def test_external_function_side_effects(self):
        if False:
            return 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    return 10
                test_self.assertEqual(name, qual_names.QN('g'))
                return (None, g)

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {str(type_anno)}

            def res_call(self, ns, types_ns, node, f_type, args, keywords):
                if False:
                    print('Hello World!')
                test_self.assertIsNone(f_type)
                return (None, {qual_names.QN('x'): {str}})

        def g():
            if False:
                while True:
                    i = 10
            pass

        def test_fn(x: int):
            if False:
                while True:
                    i = 10
            y = x
            g()
            return (x, y)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].targets[0], 'int')
        self.assertTypes(fn_body[0].value, 'int')
        self.assertTypes(fn_body[2].value.elts[0], str)
        self.assertTypes(fn_body[2].value.elts[1], 'int')

    def test_local_function_closure(self):
        if False:
            print('Hello World!')

        def test_fn(x: int):
            if False:
                print('Hello World!')

            def foo():
                if False:
                    i = 10
                    return i + 15
                return x
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[0].value, 'int')
        self.assertClosureTypes(fn_body[0], {'x': {'int'}})

    def test_local_function_closure_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x: int):
            if False:
                while True:
                    i = 10

            def foo():
                if False:
                    return 10

                def bar():
                    if False:
                        print('Hello World!')
                    return x
                bar()
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[0].body[0].value, 'int')
        self.assertClosureTypes(fn_body[0], {'x': {'int'}})
        self.assertClosureTypes(fn_body[0].body[0], {'x': {'int'}})

    def test_local_function_closure_mutable_var(self):
        if False:
            return 10

        def test_fn(x: int):
            if False:
                while True:
                    i = 10

            def foo():
                if False:
                    while True:
                        i = 10
                nonlocal x
                return x
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[1].value, 'int')
        self.assertClosureTypes(fn_body[0], {'x': {'int'}})

    def test_local_function_closure_ignored_for_bound_symbols(self):
        if False:
            return 10

        def test_fn(x: float):
            if False:
                for i in range(10):
                    print('nop')

            def foo():
                if False:
                    i = 10
                    return i + 15
                x = x + 1
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertFalse(anno.hasanno(fn_body[0].body[0].value.left, anno.Static.TYPES))
        self.assertClosureTypes(fn_body[0], {'x': {'float'}})

    def test_local_function_closure_uses_call_site_types(self):
        if False:
            print('Hello World!')

        def test_fn(x: int):
            if False:
                i = 10
                return i + 15

            def foo():
                if False:
                    while True:
                        i = 10
                return x
            x = 1.0
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[0].value, float)
        self.assertTypes(fn_body[1].targets[0], float)
        self.assertClosureTypes(fn_body[0], {'x': {float}})

    def test_local_function_hides_locals(self):
        if False:
            return 10

        def test_fn(a: int):
            if False:
                for i in range(10):
                    print('nop')

            def local_fn(v):
                if False:
                    print('Hello World!')
                a = v
                return a
            local_fn(1)
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertFalse(anno.hasanno(fn_body[0].body[0].targets[0], anno.Static.TYPES))

    def test_local_function_type(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x: int):
            if False:
                while True:
                    i = 10

            def foo() -> int:
                if False:
                    while True:
                        i = 10
                return x
            foo()
        (node, _) = TestTranspiler(BasicTestResolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value.func, Callable[[Any], int])
        self.assertTypes(fn_body[1].value, int)
        self.assertTypes(fn_body[1], int)

    def test_side_effects_on_arg_function_closure(self):
        if False:
            print('Hello World!')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_name(self, ns, types_ns, name):
                if False:
                    print('Hello World!')
                test_self.assertEqual(name, qual_names.QN('g'))
                return ({Callable[[Callable], None]}, g)

            def res_value(self, ns, value):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertEqual(value, 1.0)
                return {float}

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    i = 10
                    return i + 15
                return {str(type_anno)}

            def res_call(self, ns, types_ns, node, f_type, args, keywords):
                if False:
                    while True:
                        i = 10
                test_self.assertEqual(node.func.id, 'g')
                test_self.assertEqual(f_type, (Callable[[Callable], None],))
                return (None, {qual_names.QN('x'): {str}})

        def g(foo):
            if False:
                print('Hello World!')
            del foo
            pass

        def test_fn(x: int):
            if False:
                for i in range(10):
                    print('nop')

            def foo():
                if False:
                    return 10
                return x
            x = 1.0
            g(foo)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].body[0].value, str)

    def test_subscript(self):
        if False:
            return 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    for i in range(10):
                        print('nop')
                return {list}

            def res_value(self, ns, value):
                if False:
                    while True:
                        i = 10
                return {int}

            def res_slice(self, ns, types_ns, node, value, slice_):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertSetEqual(value, {list})
                test_self.assertSetEqual(slice_, {int})
                return {str}

        def test_fn(a):
            if False:
                print('Hello World!')
            return a[1]
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, str)
        self.assertTypes(fn_body[0].value.value, list)
        self.assertTypes(fn_body[0].value.slice, int)

    def test_tuple_unpacking(self):
        if False:
            return 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {list}

            def res_value(self, ns, value):
                if False:
                    for i in range(10):
                        print('nop')
                return {int}

            def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
                if False:
                    while True:
                        i = 10
                test_self.assertIn(node_or_slice, (0, 1))
                test_self.assertSetEqual(value, {list})
                test_self.assertSetEqual(slice_, {int})
                if node_or_slice == 0:
                    return {float}
                else:
                    return {str}

        def test_fn(t):
            if False:
                while True:
                    i = 10
            (a, b) = t
            return (a, b)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value, ((float, str),))
        self.assertTypes(fn_body[1].value.elts[0], float)
        self.assertTypes(fn_body[1].value.elts[1], str)

    def test_compare(self):
        if False:
            i = 10
            return i + 15
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    print('Hello World!')
                return {int}

            def res_compare(self, ns, types_ns, node, left, right):
                if False:
                    while True:
                        i = 10
                test_self.assertSetEqual(left, {int})
                test_self.assertListEqual(right, [{int}])
                return {bool}

        def test_fn(a, b):
            if False:
                print('Hello World!')
            return a < b
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, bool)
        self.assertTypes(fn_body[0].value.left, int)
        self.assertTypes(fn_body[0].value.comparators[0], int)

    def test_binop(self):
        if False:
            return 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    while True:
                        i = 10
                return {list}

            def res_binop(self, ns, types_ns, node, left, right):
                if False:
                    print('Hello World!')
                test_self.assertSetEqual(left, {list})
                test_self.assertSetEqual(right, {list})
                return {float}

        def test_fn(a, b):
            if False:
                while True:
                    i = 10
            return a @ b
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, float)
        self.assertTypes(fn_body[0].value.left, list)
        self.assertTypes(fn_body[0].value.right, list)

    def test_unop(self):
        if False:
            while True:
                i = 10

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {list}

            def res_unop(self, ns, types_ns, node, opnd):
                if False:
                    i = 10
                    return i + 15
                return {float}

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            return -a
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, float)
        self.assertTypes(fn_body[0].value.operand, list)

    def test_tuple_literal(self):
        if False:
            for i in range(10):
                print('nop')

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {int}

        def test_fn(a, b):
            if False:
                while True:
                    i = 10
            return (a, b)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, ((int, int),))
        self.assertTypes(fn_body[0].value.elts[0], int)
        self.assertTypes(fn_body[0].value.elts[1], int)

    def test_list_literal(self):
        if False:
            return 10

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {int}

            def res_list_literal(self, ns, elt_types):
                if False:
                    print('Hello World!')
                all_types = set()
                for s in elt_types:
                    all_types |= s
                return {List[t] for t in all_types}

        def test_fn(a, b):
            if False:
                return 10
            return [a, b]
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[0].value, List[int])
        self.assertTypes(fn_body[0].value.elts[0], int)
        self.assertTypes(fn_body[0].value.elts[1], int)

    def test_tuple_unpacking_syntactic(self):
        if False:
            while True:
                i = 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    i = 10
                    return i + 15
                if name == qual_names.QN('a'):
                    return {int}
                else:
                    return {float}

            def res_value(self, ns, value):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertIn(value, (0, 1))
                return int

            def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
                if False:
                    i = 10
                    return i + 15
                test_self.assertIn(node_or_slice, (0, 1))
                test_self.assertSetEqual(value, {(int, float)})
                test_self.assertEqual(slice_, int)
                return {t[node_or_slice] for t in value}

        def test_fn(a, b):
            if False:
                print('Hello World!')
            (c, d) = (a, b)
            return (c, d)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value, ((int, float),))
        self.assertTypes(fn_body[1].value.elts[0], int)
        self.assertTypes(fn_body[1].value.elts[1], float)

    def test_tuple_unpacking_operational(self):
        if False:
            print('Hello World!')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    return 10
                return {(int, float)}

            def res_value(self, ns, value):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertIn(value, (0, 1))
                return int

            def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
                if False:
                    while True:
                        i = 10
                test_self.assertIn(node_or_slice, (0, 1))
                test_self.assertSetEqual(value, {(int, float)})
                test_self.assertEqual(slice_, int)
                return {t[node_or_slice] for t in value}

        def test_fn(a):
            if False:
                return 10
            (c, d) = a
            return (c, d)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value, ((int, float),))
        self.assertTypes(fn_body[1].value.elts[0], int)
        self.assertTypes(fn_body[1].value.elts[1], float)

    def test_list_expansion_syntactic(self):
        if False:
            while True:
                i = 10
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    for i in range(10):
                        print('nop')
                if name == qual_names.QN('a'):
                    return {int}
                else:
                    return {float}

            def res_value(self, ns, value):
                if False:
                    print('Hello World!')
                test_self.assertIn(value, (0, 1))
                return int

            def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
                if False:
                    print('Hello World!')
                test_self.assertIn(node_or_slice, (0, 1))
                test_self.assertSetEqual(value, {(int, float)})
                test_self.assertEqual(slice_, int)
                return {t[node_or_slice] for t in value}

        def test_fn(a, b):
            if False:
                return 10
            [c, d] = (a, b)
            return (c, d)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value, ((int, float),))
        self.assertTypes(fn_body[1].value.elts[0], int)
        self.assertTypes(fn_body[1].value.elts[1], float)

    def test_list_expansion_operational(self):
        if False:
            print('Hello World!')
        test_self = self

        class Resolver(type_inference.Resolver):

            def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
                if False:
                    print('Hello World!')
                if name == qual_names.QN('a'):
                    return {int}
                else:
                    return {float}

            def res_value(self, ns, value):
                if False:
                    i = 10
                    return i + 15
                test_self.assertIn(value, (0, 1))
                return int

            def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
                if False:
                    for i in range(10):
                        print('nop')
                test_self.assertIn(node_or_slice, (0, 1))
                test_self.assertSetEqual(value, {(int, float)})
                test_self.assertEqual(slice_, int)
                return {t[node_or_slice] for t in value}

        def test_fn(a, b):
            if False:
                return 10
            [c, d] = (a, b)
            return (c, d)
        (node, _) = TestTranspiler(Resolver).transform(test_fn, None)
        fn_body = node.body
        self.assertTypes(fn_body[1].value, ((int, float),))
        self.assertTypes(fn_body[1].value.elts[0], int)
        self.assertTypes(fn_body[1].value.elts[1], float)
if __name__ == '__main__':
    test.main()
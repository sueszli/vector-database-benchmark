"""Tests for reaching_definitions module."""
import unittest
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import naming
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import reaching_definitions
global_a = 7
global_b = 17

class ReachingDefinitionsAnalyzerTestBase(unittest.TestCase):

    def _parse_and_analyze(self, test_fn):
        if False:
            print('Hello World!')
        (node, source) = parser.parse_entity(test_fn, future_features=())
        entity_info = transformer.EntityInfo(name=test_fn.__name__, source_code=source, source_file=None, future_features=(), namespace={})
        node = qual_names.resolve(node)
        namer = naming.Namer({})
        ctx = transformer.Context(entity_info, namer, None)
        node = activity.resolve(node, ctx)
        graphs = cfg.build(node)
        node = reaching_definitions.resolve(node, ctx, graphs, reaching_definitions.Definition)
        return node

    def assertHasDefs(self, node, num):
        if False:
            return 10
        defs = anno.getanno(node, anno.Static.DEFINITIONS)
        self.assertEqual(len(defs), num)
        for r in defs:
            self.assertIsInstance(r, reaching_definitions.Definition)

    def assertHasDefinedIn(self, node, expected):
        if False:
            i = 10
            return i + 15
        defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
        defined_in_str = set((str(v) for v in defined_in))
        if not expected:
            expected = ()
        if not isinstance(expected, tuple):
            expected = (expected,)
        self.assertSetEqual(defined_in_str, set(expected))

    def assertSameDef(self, first, second):
        if False:
            i = 10
            return i + 15
        self.assertHasDefs(first, 1)
        self.assertHasDefs(second, 1)
        self.assertIs(anno.getanno(first, anno.Static.DEFINITIONS)[0], anno.getanno(second, anno.Static.DEFINITIONS)[0])

    def assertNotSameDef(self, first, second):
        if False:
            return 10
        self.assertHasDefs(first, 1)
        self.assertHasDefs(second, 1)
        self.assertIsNot(anno.getanno(first, anno.Static.DEFINITIONS)[0], anno.getanno(second, anno.Static.DEFINITIONS)[0])

class ReachingDefinitionsAnalyzerTest(ReachingDefinitionsAnalyzerTestBase):

    def test_conditional(self):
        if False:
            return 10

        def test_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = []
            if b:
                a = []
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[0].targets[0], 1)
        self.assertHasDefs(fn_body[1].test, 1)
        self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value, 2)
        self.assertHasDefinedIn(fn_body[1], ('a', 'b'))

    def test_try_in_conditional(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a, b):
            if False:
                print('Hello World!')
            a = []
            if b:
                try:
                    pass
                except:
                    pass
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
        self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))

    def test_conditional_in_try_in_conditional(self):
        if False:
            while True:
                i = 10

        def test_fn(a, b):
            if False:
                while True:
                    i = 10
            a = []
            if b:
                try:
                    if b:
                        a = []
                except TestException:
                    pass
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
        self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
        self.assertHasDefinedIn(fn_body[1].body[0].body[0], ('a', 'b'))

    def test_conditional_in_except_in_conditional(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = []
            if b:
                try:
                    pass
                except TestException as e:
                    if b:
                        a = []
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
        self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
        self.assertHasDefinedIn(fn_body[1].body[0].handlers[0].body[0], ('a', 'b'))

    def test_while(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            max(a)
            while True:
                a = a
                a = a
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[0].value.args[0], 1)
        self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[1].body[1].targets[0], 1)
        self.assertHasDefs(fn_body[1].body[1].value, 1)
        self.assertHasDefs(fn_body[1].body[0].value, 2)
        self.assertHasDefs(fn_body[2].value, 2)

    def test_while_else(self):
        if False:
            while True:
                i = 10

        def test_fn(x, i):
            if False:
                print('Hello World!')
            y = 0
            while x:
                x += i
                if i:
                    break
            else:
                y = 1
            return (x, y)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[0].targets[0], 1)
        self.assertHasDefs(fn_body[1].test, 2)
        self.assertHasDefs(fn_body[1].body[0].target, 1)
        self.assertHasDefs(fn_body[1].body[1].test, 1)
        self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value.elts[0], 2)
        self.assertHasDefs(fn_body[2].value.elts[1], 2)

    def test_for_else(self):
        if False:
            while True:
                i = 10

        def test_fn(x, i):
            if False:
                while True:
                    i = 10
            y = 0
            for i in x:
                x += i
                if i:
                    break
                else:
                    continue
            else:
                y = 1
            return (x, y)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[0].targets[0], 1)
        self.assertHasDefs(fn_body[1].target, 1)
        self.assertHasDefs(fn_body[1].body[0].target, 1)
        self.assertHasDefs(fn_body[1].body[1].test, 1)
        self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value.elts[0], 2)
        self.assertHasDefs(fn_body[2].value.elts[1], 2)

    def test_nested_functions(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a, b):
            if False:
                return 10
            a = []
            if b:
                a = []

                def foo():
                    if False:
                        while True:
                            i = 10
                    return a
                foo()
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        def_of_a_in_if = fn_body[1].body[0].targets[0]
        self.assertHasDefs(fn_body[0].targets[0], 1)
        self.assertHasDefs(fn_body[1].test, 1)
        self.assertHasDefs(def_of_a_in_if, 1)
        self.assertHasDefs(fn_body[2].value, 2)
        inner_fn_body = fn_body[1].body[1].body
        def_of_a_in_foo = inner_fn_body[0].value
        self.assertHasDefs(def_of_a_in_foo, 0)

    def test_nested_functions_isolation(self):
        if False:
            print('Hello World!')

        def test_fn(a):
            if False:
                return 10
            a = 0

            def child():
                if False:
                    return 10
                a = 1
                return a
            child()
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        parent_return = fn_body[3]
        child_return = fn_body[1].body[1]
        self.assertNotSameDef(parent_return.value, child_return.value)

    def test_function_call_in_with(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(_):
            if False:
                return 10
            pass

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            with foo(a):
                return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[0].items[0].context_expr.func, 0)
        self.assertHasDefs(fn_body[0].items[0].context_expr.args[0], 1)

    def test_mutation_subscript(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                print('Hello World!')
            l = []
            l[0] = a
            return l
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        creation = fn_body[0].targets[0]
        mutation = fn_body[1].targets[0].value
        use = fn_body[2].value
        self.assertSameDef(creation, mutation)
        self.assertSameDef(creation, use)

    def test_deletion_partial(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a):
            if False:
                i = 10
                return i + 15
            a = 0
            if a:
                del a
            else:
                a = 1
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        first_def = fn_body[0].targets[0]
        second_def = fn_body[1].orelse[0].targets[0]
        use = fn_body[2].value
        self.assertNotSameDef(use, first_def)
        self.assertSameDef(use, second_def)

    def test_deletion_total(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                print('Hello World!')
            if a:
                a = 0
            else:
                a = 1
            del a
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        use = fn_body[2].value
        self.assertHasDefs(use, 0)

    def test_replacement(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a):
            if False:
                i = 10
                return i + 15
            return a

        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            a = foo(a)
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        param = node.args.args[0]
        source = fn_body[0].value.args[0]
        target = fn_body[0].targets[0]
        retval = fn_body[1].value
        self.assertSameDef(param, source)
        self.assertNotSameDef(source, target)
        self.assertSameDef(target, retval)

    def test_comprehension_leaking(self):
        if False:
            while True:
                i = 10

        def test_fn(a):
            if False:
                return 10
            _ = [x for x in a]
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        listcomp_target = fn_body[0].value.generators[0].target
        retval = fn_body[1].value
        self.assertHasDefs(retval, 0)

    def test_function_definition(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                i = 10
                return i + 15

            def a():
                if False:
                    print('Hello World!')
                pass
            if a:
                a = None
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[1].test, 1)
        self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value, 2)
        self.assertHasDefinedIn(fn_body[1], ('a',))

    def test_definitions_in_except_block(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn():
            if False:
                while True:
                    i = 10
            try:
                pass
            except ValueError:
                a = None
            if a:
                a = None
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[1].test, 1)
        self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value, 2)
        self.assertHasDefinedIn(fn_body[1], ('a',))

    def test_definitions_in_except_block_of_raising_try(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                print('Hello World!')
            try:
                raise ValueError()
            except ValueError:
                a = None
            if a:
                a = None
            return a
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[1].test, 1)
        self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[2].value, 2)
        self.assertHasDefinedIn(fn_body[1], ('a',))

    def test_global(self):
        if False:
            return 10

        def test_fn():
            if False:
                i = 10
                return i + 15
            global global_a
            global global_b
            if global_a:
                global_b = []
            return (global_a, global_b)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[2].test, 1)
        self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[1], 2)
        self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])
        self.assertHasDefinedIn(fn_body[2], ('global_a', 'global_b'))

    def test_nonlocal(self):
        if False:
            return 10
        a = 3
        b = 13

        def test_fn():
            if False:
                while True:
                    i = 10
            nonlocal a
            nonlocal b
            if a:
                b = []
            return (a, b)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[2].test, 1)
        self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[1], 2)
        self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])
        self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

    def test_nonlocal_in_nested_function(self):
        if False:
            return 10
        a = 3
        b = 13

        def test_fn():
            if False:
                return 10
            a = 3
            b = 13

            def local_fn():
                if False:
                    i = 10
                    return i + 15
                nonlocal a, b
                if a:
                    b = []
                return (a, b)
            return local_fn()
        node = self._parse_and_analyze(test_fn)
        local_body = node.body[2].body
        self.assertHasDefs(local_body[1].test, 1)
        self.assertHasDefs(local_body[1].body[0].targets[0], 1)
        self.assertHasDefs(local_body[2].value.elts[0], 1)
        self.assertHasDefs(local_body[2].value.elts[1], 2)
        self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])
        self.assertHasDefinedIn(local_body[1], ('a', 'b'))

class ReachingDefinitionsAnalyzerTestPy3(ReachingDefinitionsAnalyzerTestBase):
    """Tests which can only run in Python 3."""

    def test_nonlocal(self):
        if False:
            for i in range(10):
                print('nop')
        a = 3
        b = 13

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            nonlocal b
            if a:
                b = []
            return (a, b)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasDefs(fn_body[2].test, 1)
        self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[0], 1)
        self.assertHasDefs(fn_body[3].value.elts[1], 2)
        self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])
        self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

    def test_nonlocal_in_nested_function(self):
        if False:
            i = 10
            return i + 15
        a = 3
        b = 13

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            a = 3
            b = 13

            def local_fn():
                if False:
                    i = 10
                    return i + 15
                nonlocal a, b
                if a:
                    b = []
                return (a, b)
            return local_fn()
        node = self._parse_and_analyze(test_fn)
        local_body = node.body[2].body
        self.assertHasDefs(local_body[1].test, 1)
        self.assertHasDefs(local_body[1].body[0].targets[0], 1)
        self.assertHasDefs(local_body[2].value.elts[0], 1)
        self.assertHasDefs(local_body[2].value.elts[1], 2)
        self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])
        self.assertHasDefinedIn(local_body[1], ('a', 'b'))
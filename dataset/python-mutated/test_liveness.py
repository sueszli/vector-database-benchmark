"""Tests for liveness module."""
import unittest
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import naming
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import liveness
from nvidia.dali._autograph.pyct.static_analysis import reaching_fndefs
global_a = 7
global_b = 17

class LivenessAnalyzerTestBase(unittest.TestCase):

    def _parse_and_analyze(self, test_fn):
        if False:
            for i in range(10):
                print('nop')
        (node, source) = parser.parse_entity(test_fn, future_features=())
        entity_info = transformer.EntityInfo(name=test_fn.__name__, source_code=source, source_file=None, future_features=(), namespace={})
        node = qual_names.resolve(node)
        namer = naming.Namer({})
        ctx = transformer.Context(entity_info, namer, None)
        node = activity.resolve(node, ctx)
        graphs = cfg.build(node)
        node = reaching_fndefs.resolve(node, ctx, graphs)
        node = liveness.resolve(node, ctx, graphs)
        return node

    def assertHasLiveOut(self, node, expected):
        if False:
            print('Hello World!')
        live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
        live_out_strs = set((str(v) for v in live_out))
        if not expected:
            expected = ()
        if not isinstance(expected, tuple):
            expected = (expected,)
        self.assertSetEqual(live_out_strs, set(expected))

    def assertHasLiveIn(self, node, expected):
        if False:
            print('Hello World!')
        live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
        live_in_strs = set((str(v) for v in live_in))
        if not expected:
            expected = ()
        if not isinstance(expected, tuple):
            expected = (expected,)
        self.assertSetEqual(live_in_strs, set(expected))

class LivenessAnalyzerTest(LivenessAnalyzerTestBase):

    def test_live_out_try_block(self):
        if False:
            i = 10
            return i + 15

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                try:
                    pass
                except:
                    pass
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'x')
        self.assertHasLiveOut(fn_body[0].body[0], 'x')

    def test_live_out_if_inside_except(self):
        if False:
            return 10

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                try:
                    pass
                except:
                    if b > 0:
                        x = b
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'x')
        self.assertHasLiveOut(fn_body[0].body[0], 'x')
        self.assertHasLiveOut(fn_body[0].body[0].handlers[0].body[0], 'x')

    def test_live_out_stacked_if(self):
        if False:
            i = 10
            return i + 15

        def test_fn(x, a):
            if False:
                while True:
                    i = 10
            if a > 0:
                x = 0
            if a > 1:
                x = 1
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ('a', 'x'))
        self.assertHasLiveOut(fn_body[1], 'x')

    def test_live_out_stacked_if_else(self):
        if False:
            while True:
                i = 10

        def test_fn(x, a):
            if False:
                for i in range(10):
                    print('nop')
            if a > 0:
                x = 0
            if a > 1:
                x = 1
            else:
                x = 2
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'a')
        self.assertHasLiveOut(fn_body[1], 'x')

    def test_live_out_for_basic(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a):
            if False:
                i = 10
                return i + 15
            for i in range(a):
                x += i
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'x')

    def test_live_out_for_iterate(self):
        if False:
            print('Hello World!')

        def test_fn(x, a):
            if False:
                while True:
                    i = 10
            for i in range(a):
                x += i
            return (x, i)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ('x', 'i'))

    def test_live_out_attributes(self):
        if False:
            print('Hello World!')

        def test_fn(x, a):
            if False:
                i = 10
                return i + 15
            if a > 0:
                x.y = 0
            return x.y
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ('x.y', 'x'))

    def test_live_out_nested_functions(self):
        if False:
            while True:
                i = 10

        def test_fn(a, b):
            if False:
                return 10
            if b:
                a = []

            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                return a
            foo()
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'a')

    def test_live_out_nested_functions_defined_ahead(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(a, b):
            if False:
                i = 10
                return i + 15

            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                return a
            if b:
                a = []
            return foo
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[1], ('a', 'foo'))

    def test_live_out_nested_functions_defined_after(self):
        if False:
            i = 10
            return i + 15

        def test_fn(a, b):
            if False:
                return 10
            if b:
                a = []

            def foo():
                if False:
                    while True:
                        i = 10
                return a
            return foo
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ('a',))

    def test_live_out_lambda(self):
        if False:
            print('Hello World!')

        def test_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if b:
                a = []
            foo = lambda : a
            if b:
                pass
            return foo
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ('a', 'b'))
        self.assertHasLiveOut(fn_body[2], ('foo',))

    def test_live_out_nested_functions_hidden_by_argument(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(b):
            if False:
                while True:
                    i = 10

            def foo(a):
                if False:
                    return 10
                return a
            if b:
                a = []
            return foo
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[1], 'foo')

    def test_live_out_nested_functions_isolation(self):
        if False:
            return 10

        def test_fn(b):
            if False:
                i = 10
                return i + 15
            if b:
                a = 0

            def child():
                if False:
                    for i in range(10):
                        print('nop')
                max(a)
                a = 1
                return a
            child()
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], 'max')

    def test_live_out_deletion(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, y, a):
            if False:
                while True:
                    i = 10
            for _ in a:
                if x:
                    del y
                else:
                    y = 0
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[0], ())

    def test_live_in_pass(self):
        if False:
            i = 10
            return i + 15

        def test_fn(x, a, b, c):
            if False:
                print('Hello World!')
            if a > 0:
                pass
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('x',))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_in_raise(self):
        if False:
            return 10

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                b = b + 1
                raise c
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('b', 'c'))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_out_except_variable(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a):
            if False:
                print('Hello World!')
            try:
                pass
            except a as b:
                raise b
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('b', 'x'))

    def test_live_in_return_statement(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            if a > 0:
                return x
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('x',))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_in_try_block(self):
        if False:
            while True:
                i = 10

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                try:
                    pass
                except:
                    pass
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('x',))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_in_try_orelse(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                try:
                    pass
                except:
                    pass
                else:
                    x = b
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'b', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('b', 'x'))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_in_if_inside_except(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a, b, c):
            if False:
                while True:
                    i = 10
            if a > 0:
                try:
                    pass
                except:
                    if b > 0:
                        x = b
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'b', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0], ('b', 'x'))
        self.assertHasLiveIn(fn_body[0].body[0].handlers[0].body[0], ('b', 'x'))
        self.assertHasLiveIn(fn_body[1], ('x',))

    def test_live_in_stacked_if(self):
        if False:
            return 10

        def test_fn(x, a, b, c):
            if False:
                return 10
            if a > 0:
                x = b
            if c > 1:
                x = 0
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'x'))
        self.assertHasLiveIn(fn_body[1], ('c', 'x'))

    def test_live_in_stacked_if_else(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, a, b, c, d):
            if False:
                return 10
            if a > 1:
                x = b
            else:
                x = c
            if d > 0:
                x = 0
            return x
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'd'))
        self.assertHasLiveIn(fn_body[1], ('d', 'x'))

    def test_live_in_for_basic(self):
        if False:
            return 10

        def test_fn(x, y, a):
            if False:
                return 10
            for i in a:
                x = i
                y += x
                z = 0
            return (y, z)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

    def test_live_in_for_nested(self):
        if False:
            print('Hello World!')

        def test_fn(x, y, a):
            if False:
                print('Hello World!')
            for i in a:
                for j in i:
                    x = i
                    y += x
                    z = j
            return (y, z)
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

    def test_live_in_deletion(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(x, y, a):
            if False:
                for i in range(10):
                    print('nop')
            for _ in a:
                if x:
                    del y
                else:
                    y = 0
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('a', 'x', 'y'))

    def test_live_in_generator_comprehension(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(y):
            if False:
                i = 10
                return i + 15
            if all((x for x in y)):
                return
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('all', 'y'))

    def test_live_in_list_comprehension(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(y):
            if False:
                print('Hello World!')
            if [x for x in y]:
                return
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('y',))

    def test_live_in_list_comprehension_expression(self):
        if False:
            return 10

        def test_fn(y, s):
            if False:
                i = 10
                return i + 15
            s += foo([x for x in y])
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('y', 'foo', 's'))

    def test_live_in_set_comprehension(self):
        if False:
            for i in range(10):
                print('nop')

        def test_fn(y):
            if False:
                return 10
            if {x for x in y}:
                return
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('y',))

    def test_live_in_dict_comprehension(self):
        if False:
            while True:
                i = 10

        def test_fn(y):
            if False:
                for i in range(10):
                    print('nop')
            if {k: v for (k, v) in y}:
                return
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveIn(fn_body[0], ('y',))

    def test_global_symbol(self):
        if False:
            return 10

        def test_fn(c):
            if False:
                i = 10
                return i + 15
            global global_a
            global global_b
            if global_a:
                global_b = c
            else:
                global_b = c
            return global_b
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[2], ('global_b',))
        self.assertHasLiveIn(fn_body[2], ('global_a', 'c'))

    def test_nonlocal_symbol(self):
        if False:
            i = 10
            return i + 15
        nonlocal_a = 3
        nonlocal_b = 13

        def test_fn(c):
            if False:
                while True:
                    i = 10
            nonlocal nonlocal_a
            nonlocal nonlocal_b
            if nonlocal_a:
                nonlocal_b = c
            else:
                nonlocal_b = c
            return nonlocal_b
        node = self._parse_and_analyze(test_fn)
        fn_body = node.body
        self.assertHasLiveOut(fn_body[2], ('nonlocal_b',))
        self.assertHasLiveIn(fn_body[2], ('nonlocal_a', 'c'))
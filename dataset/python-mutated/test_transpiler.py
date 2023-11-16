"""Tests for transpiler module."""
import threading
import unittest
import gast
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct import transpiler

class FlipSignTransformer(transformer.Base):

    def visit_BinOp(self, node):
        if False:
            print('Hello World!')
        if isinstance(node.op, gast.Add):
            node.op = gast.Sub()
        return self.generic_visit(node)

class TestTranspiler(transpiler.PyToPy):

    def get_caching_key(self, ctx):
        if False:
            print('Hello World!')
        del ctx
        return 0

    def get_extra_locals(self):
        if False:
            while True:
                i = 10
        return {}

    def transform_ast(self, node, ctx):
        if False:
            for i in range(10):
                print('nop')
        return FlipSignTransformer(ctx).visit(node)
global_var_for_test_global = 1
global_var_for_test_namespace_collisions = object()

class PyToPyTest(unittest.TestCase):

    def test_basic(self):
        if False:
            return 10

        def f(a):
            if False:
                i = 10
                return i + 15
            return a + 1
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 0)

    def test_closure(self):
        if False:
            return 10
        b = 1

        def f(a):
            if False:
                while True:
                    i = 10
            return a + b
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 0)
        b = 2
        self.assertEqual(f(1), -1)

    def test_global(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a + global_var_for_test_global
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        global global_var_for_test_global
        global_var_for_test_global = 1
        self.assertEqual(f(1), 0)
        global_var_for_test_global = 2
        self.assertEqual(f(1), -1)

    def test_defaults(self):
        if False:
            while True:
                i = 10
        b = 2
        c = 1

        def f(a, d=c + 1):
            if False:
                return 10
            return a + b + d
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 1 - 2 - 2)
        c = 0
        self.assertEqual(f(1), 1 - 2 - 2)
        b = 1
        self.assertEqual(f(1), 1 - 2 - 1)

    def test_call_tree(self):
        if False:
            return 10

        def g(a):
            if False:
                i = 10
                return i + 15
            return a + 1

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return g(a) + 1
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 1 - 1 + 1)

    def test_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        b = 2
        f = lambda x: b + (x if x > 0 else -x)
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 2 - 1)
        self.assertEqual(f(-1), 2 - 1)
        b = 3
        self.assertEqual(f(1), 3 - 1)
        self.assertEqual(f(-1), 3 - 1)

    def test_multiple_lambdas(self):
        if False:
            i = 10
            return i + 15
        (a, b) = (1, 2)
        (f, _) = (lambda x: a + x, lambda y: b * y)
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 1 - 1)

    def test_nested_functions(self):
        if False:
            while True:
                i = 10
        b = 2

        def f(x):
            if False:
                print('Hello World!')

            def g(x):
                if False:
                    while True:
                        i = 10
                return b + x
            return g(x)
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 2 - 1)

    def test_nested_lambda(self):
        if False:
            i = 10
            return i + 15
        b = 2

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            g = lambda x: b + x
            return g(x)
        tr = TestTranspiler()
        (f, _, _) = tr.transform(f, None)
        self.assertEqual(f(1), 2 - 1)

    def test_concurrency(self):
        if False:
            return 10

        def f():
            if False:
                return 10
            pass
        outputs = []
        tr = TestTranspiler()
        assert tr.get_caching_key(None) == tr.get_caching_key(None)

        def conversion_thread():
            if False:
                return 10
            (_, mod, _) = tr.transform(f, None)
            outputs.append(mod.__name__)
        threads = tuple((threading.Thread(target=conversion_thread) for _ in range(10)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(outputs)), 1)

    def test_reentrance(self):
        if False:
            print('Hello World!')

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            return 1 + 1

        class ReentrantTranspiler(transpiler.PyToPy):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(ReentrantTranspiler, self).__init__()
                self._recursion_depth = 0

            def get_caching_key(self, ctx):
                if False:
                    while True:
                        i = 10
                del ctx
                return 0

            def get_extra_locals(self):
                if False:
                    i = 10
                    return i + 15
                return {}

            def transform_ast(self, node, ctx):
                if False:
                    return 10
                self._recursion_depth += 1
                if self._recursion_depth < 2:
                    self.transform(test_fn, None)
                return FlipSignTransformer(ctx).visit(node)
        tr = ReentrantTranspiler()
        (f, _, _) = tr.transform(test_fn, None)
        self.assertEqual(f(), 0)

    def test_namespace_collisions_avoided(self):
        if False:
            for i in range(10):
                print('nop')

        class TestClass(object):

            def global_var_for_test_namespace_collisions(self):
                if False:
                    for i in range(10):
                        print('nop')
                return global_var_for_test_namespace_collisions
        tr = TestTranspiler()
        obj = TestClass()
        (f, _, _) = tr.transform(obj.global_var_for_test_namespace_collisions, None)
        self.assertIs(f(obj), global_var_for_test_namespace_collisions)
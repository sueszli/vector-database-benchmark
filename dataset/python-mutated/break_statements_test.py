"""Tests for break_statements module."""
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test

class BreakCanonicalizationTest(converter_testing.TestCase):

    def assertTransformedEquivalent(self, f, *inputs):
        if False:
            i = 10
            return i + 15
        tr = self.transform(f, break_statements)
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_while_loop(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            v = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    break
                v.append(x)
            return v
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 1)
        self.assertTransformedEquivalent(f, 4)

    def test_while_loop_preserves_directives(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                i = 10
                return i + 15
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    break
        (_, node, ctx) = self.transform(f, (), include_ast=True)
        fake_annotation = object()
        anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
        node = break_statements.transform(node, ctx)
        self.assertIs(anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

    def test_for_loop(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            v = []
            for x in a:
                x -= 1
                if x % 2 == 0:
                    break
                v.append(x)
            return v
        tr = self.transform(f, break_statements)
        self.assertEqual([3], tr([5, 4]))

    def test_for_loop_preserves_directives(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            for x in a:
                if x % 2 == 0:
                    break
        (_, node, ctx) = self.transform(f, (), include_ast=True)
        fake_annotation = object()
        anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
        node = break_statements.transform(node, ctx)
        self.assertIs(anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

    def test_nested(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            v = []
            u = []
            w = []
            while x > 0:
                x -= 1
                if x % 2 == 0:
                    if x % 3 != 0:
                        u.append(x)
                    else:
                        w.append(x)
                        break
                v.append(x)
            return (v, u, w)
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 11)

    def test_nested_loops(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                print('Hello World!')
            v = []
            u = []
            while x > 0:
                x -= 1
                y = x
                while y > 0:
                    y -= 1
                    if y % 2 == 0:
                        break
                    u.append(y)
                if x == 0:
                    break
                v.append(x)
            return (v, u)
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, 3)
        self.assertTransformedEquivalent(f, 5)

    def test_loop_orelse(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            v = []
            u = []
            while x > 0:
                x -= 1
                y = x
                while y > 1:
                    break
                else:
                    u.append(y)
                    break
                v.append(x)
            return (v, u)
        self.assertTransformedEquivalent(f, 0)
        self.assertTransformedEquivalent(f, 2)
        self.assertTransformedEquivalent(f, 3)

    def test_multiple_correlated_breaks_with_side_effects(self):
        if False:
            while True:
                i = 10

        def f(cond1):
            if False:
                print('Hello World!')
            lst = []
            while True:
                if cond1:
                    lst.append(1)
                else:
                    break
                if lst[-1] > 0:
                    break
            return lst
        self.assertTransformedEquivalent(f, True)
        self.assertTransformedEquivalent(f, False)
if __name__ == '__main__':
    test.main()
"""Tests for anf module."""
import textwrap
import gast
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.common_transformers import anf
from tensorflow.python.platform import test

def exec_test_function():
    if False:
        for i in range(10):
            print('nop')
    exec('computed' + 5 + 'stuff', globals(), locals())

def exec_expected_result():
    if False:
        i = 10
        return i + 15
    tmp_1001 = 'computed' + 5
    tmp_1002 = tmp_1001 + 'stuff'
    tmp_1003 = globals()
    tmp_1004 = locals()
    exec(tmp_1002, tmp_1003, tmp_1004)

class AnfTestBase(test.TestCase):

    def _simple_context(self):
        if False:
            i = 10
            return i + 15
        entity_info = transformer.EntityInfo(name='test_fn', source_code=None, source_file=None, future_features=(), namespace=None)
        return transformer.Context(entity_info, None, None)

    def assert_same_ast(self, expected_node, node, msg=None):
        if False:
            while True:
                i = 10
        expected_source = parser.unparse(expected_node, indentation='  ')
        expected_str = textwrap.dedent(expected_source).strip()
        got_source = parser.unparse(node, indentation='  ')
        got_str = textwrap.dedent(got_source).strip()
        self.assertEqual(expected_str, got_str, msg=msg)

    def assert_body_anfs_as_expected(self, expected_fn, test_fn, config=None):
        if False:
            for i in range(10):
                print('nop')
        (exp_node, _) = parser.parse_entity(expected_fn, future_features=())
        (node, _) = parser.parse_entity(test_fn, future_features=())
        node = anf.transform(node, self._simple_context(), config=config)
        exp_name = exp_node.name
        node.name = exp_name
        self.assert_same_ast(exp_node, node)
        node_repeated = anf.transform(node, self._simple_context())
        self.assert_same_ast(node_repeated, node)

class AnfTransformerTest(AnfTestBase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15

        def test_function():
            if False:
                for i in range(10):
                    print('nop')
            a = 0
            return a
        (node, _) = parser.parse_entity(test_function, future_features=())
        node = anf.transform(node, self._simple_context())
        (result, _, _) = loader.load_ast(node)
        self.assertEqual(test_function(), result.test_function())

    def test_binop_basic(self):
        if False:
            print('Hello World!')

        def test_function(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            a = x + y + z
            return a

        def expected_result(x, y, z):
            if False:
                while True:
                    i = 10
            tmp_1001 = x + y
            a = tmp_1001 + z
            return a
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_if_basic(self):
        if False:
            for i in range(10):
                print('nop')

        def test_function(a, b, c, e, f, g):
            if False:
                i = 10
                return i + 15
            if a + b + c:
                d = e + f + g
                return d

        def expected_result(a, b, c, e, f, g):
            if False:
                return 10
            tmp_1001 = a + b
            tmp_1002 = tmp_1001 + c
            if tmp_1002:
                tmp_1003 = e + f
                d = tmp_1003 + g
                return d
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_nested_binop_and_return(self):
        if False:
            i = 10
            return i + 15

        def test_function(b, c, d, e):
            if False:
                for i in range(10):
                    print('nop')
            return 2 * b + c + (d + e)

        def expected_result(b, c, d, e):
            if False:
                return 10
            tmp_1001 = 2 * b
            tmp_1002 = tmp_1001 + c
            tmp_1003 = d + e
            tmp_1004 = tmp_1002 + tmp_1003
            return tmp_1004
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_function_call_and_expr(self):
        if False:
            return 10

        def test_function(call_something, a, b, y, z, c, d, e, f, g, h, i):
            if False:
                return 10
            call_something(a + b, y * z, *e + f, kwarg=c + d, **g + h + i)

        def expected_result(call_something, a, b, y, z, c, d, e, f, g, h, i):
            if False:
                while True:
                    i = 10
            tmp_1001 = g + h
            tmp_1002 = a + b
            tmp_1003 = y * z
            tmp_1004 = e + f
            tmp_1005 = c + d
            tmp_1006 = tmp_1001 + i
            call_something(tmp_1002, tmp_1003, *tmp_1004, kwarg=tmp_1005, **tmp_1006)
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_with_and_print(self):
        if False:
            print('Hello World!')

        def test_function(a, b, c):
            if False:
                i = 10
                return i + 15
            with a + b + c as d:
                print(2 * d + 1)

        def expected_result(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            tmp_1001 = a + b
            tmp_1002 = tmp_1001 + c
            with tmp_1002 as d:
                tmp_1003 = 2 * d
                tmp_1004 = tmp_1003 + 1
                print(tmp_1004)
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_nested_multi_value_assign(self):
        if False:
            i = 10
            return i + 15

        def test_function(a, b, c):
            if False:
                while True:
                    i = 10
            (x, y) = (a, a + b)
            ((z, y), x) = ((c, y + b), x + a)
            return (z, (y, x))

        def expected_result(a, b, c):
            if False:
                while True:
                    i = 10
            tmp_1001 = a + b
            (x, y) = (a, tmp_1001)
            tmp_1002 = y + b
            tmp_1003 = (c, tmp_1002)
            tmp_1004 = x + a
            ((z, y), x) = (tmp_1003, tmp_1004)
            tmp_1005 = (y, x)
            tmp_1006 = (z, tmp_1005)
            return tmp_1006
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_deeply_nested_multi_value_assign(self):
        if False:
            i = 10
            return i + 15

        def test_function(a):
            if False:
                for i in range(10):
                    print('nop')
            [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
            return [([(b, c), [d, e]], (f, g)), [(h, i, j), k]]

        def expected_result(a):
            if False:
                return 10
            [([(b, c), [d, e]], (f, g)), [(h, i, j), k]] = a
            tmp_1001 = (b, c)
            tmp_1002 = [d, e]
            tmp_1003 = [tmp_1001, tmp_1002]
            tmp_1004 = (f, g)
            tmp_1005 = (h, i, j)
            tmp_1006 = (tmp_1003, tmp_1004)
            tmp_1007 = [tmp_1005, k]
            tmp_1008 = [tmp_1006, tmp_1007]
            return tmp_1008
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_local_definition_and_binary_compare(self):
        if False:
            i = 10
            return i + 15

        def test_function():
            if False:
                print('Hello World!')

            def foo(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return 2 * a < b
            return foo

        def expected_result():
            if False:
                print('Hello World!')

            def foo(a, b):
                if False:
                    return 10
                tmp_1001 = 2 * a
                tmp_1002 = tmp_1001 < b
                return tmp_1002
            return foo
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_list_literal(self):
        if False:
            while True:
                i = 10

        def test_function(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            return [a + b, c + d, e + f]

        def expected_result(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            tmp_1001 = a + b
            tmp_1002 = c + d
            tmp_1003 = e + f
            tmp_1004 = [tmp_1001, tmp_1002, tmp_1003]
            return tmp_1004
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_tuple_literal_and_unary(self):
        if False:
            while True:
                i = 10

        def test_function(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            return (a + b, -(c + d), e + f)

        def expected_result(a, b, c, d, e, f):
            if False:
                return 10
            tmp_1001 = c + d
            tmp_1002 = a + b
            tmp_1003 = -tmp_1001
            tmp_1004 = e + f
            tmp_1005 = (tmp_1002, tmp_1003, tmp_1004)
            return tmp_1005
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_set_literal(self):
        if False:
            return 10

        def test_function(a, b, c, d, e, f):
            if False:
                for i in range(10):
                    print('nop')
            return set(a + b, c + d, e + f)

        def expected_result(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            tmp_1001 = a + b
            tmp_1002 = c + d
            tmp_1003 = e + f
            tmp_1004 = set(tmp_1001, tmp_1002, tmp_1003)
            return tmp_1004
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_dict_literal_and_repr(self):
        if False:
            i = 10
            return i + 15

        def test_function(foo, bar, baz):
            if False:
                while True:
                    i = 10
            return repr({foo + bar + baz: 7 | 8})

        def expected_result(foo, bar, baz):
            if False:
                return 10
            tmp_1001 = foo + bar
            tmp_1002 = tmp_1001 + baz
            tmp_1003 = 7 | 8
            tmp_1004 = {tmp_1002: tmp_1003}
            tmp_1005 = repr(tmp_1004)
            return tmp_1005
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_field_read_and_write(self):
        if False:
            return 10

        def test_function(a, d):
            if False:
                print('Hello World!')
            a.b.c = d.e.f + 3

        def expected_result(a, d):
            if False:
                for i in range(10):
                    print('nop')
            tmp_1001 = a.b
            tmp_1002 = d.e
            tmp_1003 = tmp_1002.f
            tmp_1001.c = tmp_1003 + 3
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_subscript_read_and_write(self):
        if False:
            for i in range(10):
                print('nop')

        def test_function(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            a[b][c] = d[e][f] + 3

        def expected_result(a, b, c, d, e, f):
            if False:
                print('Hello World!')
            tmp_1001 = a[b]
            tmp_1002 = d[e]
            tmp_1003 = tmp_1002[f]
            tmp_1001[c] = tmp_1003 + 3
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_augassign_and_delete(self):
        if False:
            return 10

        def test_function(a, x, y, z):
            if False:
                return 10
            a += x + y + z
            del a
            del z[y][x]

        def expected_result(a, x, y, z):
            if False:
                return 10
            tmp_1001 = x + y
            a += tmp_1001 + z
            del a
            tmp_1002 = z[y]
            del tmp_1002[x]
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_raise_yield_and_raise(self):
        if False:
            i = 10
            return i + 15

        def test_function(a, c, some_computed, exception):
            if False:
                for i in range(10):
                    print('nop')
            yield (a ** c)
            raise some_computed('complicated' + exception)

        def expected_result(a, c, some_computed, exception):
            if False:
                print('Hello World!')
            tmp_1001 = a ** c
            yield tmp_1001
            tmp_1002 = 'complicated' + exception
            tmp_1003 = some_computed(tmp_1002)
            raise tmp_1003
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_with_and_if_with_expressions(self):
        if False:
            return 10

        def test_function(foo, bar, function, quux, quozzle, w, x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            with foo + bar:
                function(x + y)
            if quux + quozzle:
                function(z / w)

        def expected_result(foo, bar, function, quux, quozzle, w, x, y, z):
            if False:
                while True:
                    i = 10
            tmp_1001 = foo + bar
            with tmp_1001:
                tmp_1002 = x + y
                function(tmp_1002)
            tmp_1003 = quux + quozzle
            if tmp_1003:
                tmp_1004 = z / w
                function(tmp_1004)
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_exec(self):
        if False:
            while True:
                i = 10
        self.assert_body_anfs_as_expected(exec_expected_result, exec_test_function)

    def test_simple_while_and_assert(self):
        if False:
            while True:
                i = 10

        def test_function(foo, quux):
            if False:
                print('Hello World!')
            while foo:
                assert quux
                foo = foo + 1 * 3

        def expected_result(foo, quux):
            if False:
                return 10
            while foo:
                assert quux
                tmp_1001 = 1 * 3
                foo = foo + tmp_1001
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_for(self):
        if False:
            while True:
                i = 10

        def test_function(compute, something, complicated, foo):
            if False:
                for i in range(10):
                    print('nop')
            for foo in compute(something + complicated):
                bar = foo + 1 * 3
            return bar

        def expected_result(compute, something, complicated, foo):
            if False:
                return 10
            tmp_1001 = something + complicated
            tmp_1002 = compute(tmp_1001)
            for foo in tmp_1002:
                tmp_1003 = 1 * 3
                bar = foo + tmp_1003
            return bar
        self.assert_body_anfs_as_expected(expected_result, test_function)

    def test_controversial(self):
        if False:
            return 10

        def test_function(b, c, d, f):
            if False:
                while True:
                    i = 10
            a = c + d
            a.b = c + d
            a[b] = c + d
            a += c + d
            (a, b) = c
            (a, b) = (c, d)
            a = f(c)
            a = f(c + d)
            a[b + d] = f.e(c + d)

        def expected_result(b, c, d, f):
            if False:
                for i in range(10):
                    print('nop')
            a = c + d
            a.b = c + d
            a[b] = c + d
            a += c + d
            (a, b) = c
            (a, b) = (c, d)
            a = f(c)
            tmp_1001 = c + d
            a = f(tmp_1001)
            tmp_1002 = b + d
            tmp_1003 = f.e
            tmp_1004 = c + d
            a[tmp_1002] = tmp_1003(tmp_1004)
        self.assert_body_anfs_as_expected(expected_result, test_function)

class AnfNonTransformationTest(AnfTransformerTest):
    """Test that specifying "no transformation" does nothing.

  Reuses all the examples of AnfTransformerTest by overriding
  `assert_body_anfs_as_expected_`.
  """

    def assert_body_anfs_as_expected(self, expected_fn, test_fn):
        if False:
            for i in range(10):
                print('nop')
        (node, _) = parser.parse_entity(test_fn, future_features=())
        orig_source = parser.unparse(node, indentation='  ')
        orig_str = textwrap.dedent(orig_source).strip()
        config = [(anf.ANY, anf.LEAVE)]
        node = anf.transform(node, self._simple_context(), config=config)
        new_source = parser.unparse(node, indentation='  ')
        new_str = textwrap.dedent(new_source).strip()
        self.assertEqual(orig_str, new_str)

class AnfConfiguredTest(AnfTestBase):

    def test_constants_in_function_calls(self):
        if False:
            while True:
                i = 10
        try:
            literals = (gast.Num, gast.Str, gast.Bytes, gast.NameConstant, gast.Name)
        except AttributeError:
            literals = (gast.Constant, gast.Name)
        config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, literals), anf.REPLACE)]

        def test_function(x, frob):
            if False:
                for i in range(10):
                    print('nop')
            return frob(x, x + 1, 2)

        def expected_result(x, frob):
            if False:
                return 10
            tmp_1001 = 2
            return frob(x, x + 1, tmp_1001)
        self.assert_body_anfs_as_expected(expected_result, test_function, config)

    def test_anf_some_function_calls(self):
        if False:
            print('Hello World!')
        allowlist = ['foo']

        def transform(parent, field, child):
            if False:
                for i in range(10):
                    print('nop')
            del field
            del child
            func_name = parent.func.id
            return str(func_name) in allowlist
        config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, anf.ANY), transform)]

        def test_function(x, foo, bar):
            if False:
                i = 10
                return i + 15
            y = foo(x, x + 1, 2)
            return bar(y, y + 1, 2)

        def expected_result(x, foo, bar):
            if False:
                print('Hello World!')
            tmp_1001 = x + 1
            tmp_1002 = 2
            y = foo(x, tmp_1001, tmp_1002)
            return bar(y, y + 1, 2)
        self.assert_body_anfs_as_expected(expected_result, test_function, config)

    def test_touching_name_constant(self):
        if False:
            print('Hello World!')
        specials = (gast.Name, gast.Constant)
        config = [(anf.ASTEdgePattern(gast.Call, anf.ANY, specials), anf.REPLACE)]

        def test_function(f):
            if False:
                print('Hello World!')
            return f(True, False, None)

        def expected_result(f):
            if False:
                return 10
            tmp_1001 = True
            tmp_1002 = False
            tmp_1003 = None
            return f(tmp_1001, tmp_1002, tmp_1003)
        self.assert_body_anfs_as_expected(expected_result, test_function, config)
if __name__ == '__main__':
    test.main()
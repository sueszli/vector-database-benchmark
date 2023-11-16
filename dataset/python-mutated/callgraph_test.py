from pytype import config
from pytype.tests import test_base
from pytype.tests import test_utils
from pytype.tools.xref import indexer

@test_base.skip(reason='The callgraph code only works in Python 3.5-6.')
class CallgraphTest(test_base.BaseTest):
    """Tests for the callgraph."""

    def index_code(self, code, **kwargs):
        if False:
            while True:
                i = 10
        'Generate references from a code string.'
        args = {'version': self.python_version}
        args.update(kwargs)
        with test_utils.Tempdir() as d:
            d.create_file('t.py', code)
            options = config.Options.create(d['t.py'])
            options.tweak(**args)
            return indexer.process_file(options, generate_callgraphs=True)

    def assertAttrsEqual(self, attrs, expected):
        if False:
            print('Hello World!')
        actual = {(x.name, x.type, x.attrib) for x in attrs}
        self.assertCountEqual(actual, expected)

    def assertCallsEqual(self, calls, expected):
        if False:
            i = 10
            return i + 15
        actual = []
        for c in calls:
            actual.append((c.function_id, [(a.name, a.node_type, a.type) for a in c.args]))
        self.assertCountEqual(actual, expected)

    def assertParamsEqual(self, params, expected):
        if False:
            return 10
        actual = {(x.name, x.type) for x in params}
        self.assertCountEqual(actual, expected)

    def assertHasFunctions(self, fns, expected):
        if False:
            while True:
                i = 10
        actual = fns.keys()
        expected = ['module'] + [f'module.{x}' for x in expected]
        self.assertCountEqual(actual, expected)

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ix = self.index_code('\n        def f(x: str):\n          y = x.strip()\n          return y\n\n        def g(y):\n          a = f(y)\n          b = complex(1, 2)\n          c = b.real\n          return c\n    ')
        fns = ix.function_map
        self.assertHasFunctions(fns, ['f', 'g'])
        f = fns['module.f']
        self.assertAttrsEqual(f.param_attrs, {('x', 'builtins.str', 'x.strip')})
        self.assertAttrsEqual(f.local_attrs, set())
        self.assertCallsEqual(f.calls, [('str.strip', [])])
        self.assertEqual(f.ret.id, 'module.f.y')
        self.assertParamsEqual(f.params, [('x', 'builtins.str')])
        g = fns['module.g']
        self.assertAttrsEqual(g.param_attrs, set())
        self.assertAttrsEqual(g.local_attrs, {('b', 'builtins.complex', 'b.real')})
        self.assertCallsEqual(g.calls, [('f', [('y', 'Param', 'typing.Any')]), ('complex', [])])
        self.assertEqual(g.ret.id, 'module.g.c')
        self.assertParamsEqual(g.params, [('y', 'typing.Any')])

    def test_remote(self):
        if False:
            return 10
        code = '\n        import foo\n\n        def f(a, b):\n          x = foo.X(a)\n          y = foo.Y(a, b)\n          z = y.bar()\n    '
        stub = '\n      class X:\n        def __init__(a: str) -> None: ...\n      class Y:\n        def __init__(a: str, b: int) -> None: ...\n        def bar() -> int: ...\n    '
        with test_utils.Tempdir() as d:
            d.create_file('t.py', code)
            d.create_file('foo.pyi', stub)
            options = config.Options.create(d['t.py'], pythonpath=d.path, version=self.python_version)
            ix = indexer.process_file(options, generate_callgraphs=True)
        fns = ix.function_map
        self.assertHasFunctions(fns, ['f'])
        f = fns['module.f']
        self.assertAttrsEqual(f.param_attrs, [])
        self.assertAttrsEqual(f.local_attrs, [('y', 'foo.Y', 'y.bar')])
        self.assertCallsEqual(f.calls, [('X', [('a', 'Param', 'typing.Any')]), ('Y', [('a', 'Param', 'typing.Any'), ('b', 'Param', 'typing.Any')]), ('Y.bar', [])])

    def test_no_outgoing_calls(self):
        if False:
            i = 10
            return i + 15
        'Capture a function with no outgoing calls.'
        ix = self.index_code('\n        def f(x: int):\n          return "hello"\n    ')
        fns = ix.function_map
        self.assertHasFunctions(fns, ['f'])
        f = fns['module.f']
        self.assertAttrsEqual(f.param_attrs, [])
        self.assertAttrsEqual(f.local_attrs, [])
        self.assertCallsEqual(f.calls, [])
        self.assertParamsEqual(f.params, [('x', 'builtins.int')])

    def test_call_records(self):
        if False:
            for i in range(10):
                print('nop')
        "Use a function's call records to infer param types."
        ix = self.index_code('\n        class A:\n          def foo(self, x):\n            return x + "1"\n\n        def f(x, y):\n          z = x + y\n          return z\n\n        def g(a):\n          return f(a, 3)\n\n        def h(b):\n          y = b\n          return y\n\n        x = g(10)\n        y = A()\n        p = h(y)\n        q = h("hello")\n        a = y.foo("1")\n    ')
        fns = ix.function_map
        self.assertHasFunctions(fns, ['A.foo', 'f', 'g', 'h'])
        expected = [('f', [('x', 'builtins.int'), ('y', 'builtins.int')]), ('g', [('a', 'builtins.int')]), ('h', [('b', 'Union[A, builtins.str]')]), ('A.foo', [('self', 'A'), ('x', 'builtins.str')])]
        for (fn, params) in expected:
            f = fns[f'module.{fn}']
            self.assertParamsEqual(f.params, params)

    def test_toplevel_calls(self):
        if False:
            for i in range(10):
                print('nop')
        "Don't index calls outside a function."
        ix = self.index_code('\n        def f(x: int):\n          return "hello"\n\n        a = f(10)\n        a.upcase()\n    ')
        fns = ix.function_map
        self.assertHasFunctions(fns, ['f'])

    def test_class_level_calls(self):
        if False:
            while True:
                i = 10
        "Don't index calls outside a function."
        ix = self.index_code('\n        def f(x: int):\n          return "hello"\n\n        class A:\n          a = f(10)\n          b = a.upcase()\n    ')
        fns = ix.function_map
        self.assertHasFunctions(fns, ['f'])
if __name__ == '__main__':
    test_base.main()
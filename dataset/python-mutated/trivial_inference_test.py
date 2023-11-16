"""Tests for apache_beam.typehints.trivial_inference."""
import types
import unittest
import apache_beam as beam
from apache_beam.typehints import row_type
from apache_beam.typehints import trivial_inference
from apache_beam.typehints import typehints
from apache_beam.utils import python_callable
global_int = 1

class TrivialInferenceTest(unittest.TestCase):

    def assertReturnType(self, expected, f, inputs=(), depth=5):
        if False:
            print('Hello World!')
        self.assertEqual(expected, trivial_inference.infer_return_type(f, inputs, debug=True, depth=depth))

    def testJumpOffsets(self):
        if False:
            print('Hello World!')
        fn = lambda x: False
        wrapper = lambda x, *args, **kwargs: [x] if fn(x, *args, **kwargs) else []
        self.assertReturnType(typehints.List[int], wrapper, [int])

    def testBuildListUnpack(self):
        if False:
            i = 10
            return i + 15
        self.assertReturnType(typehints.List[int], lambda _list: [*_list, *_list, *_list], [typehints.List[int]])

    def testBuildTupleUnpack(self):
        if False:
            print('Hello World!')
        self.assertReturnType(typehints.Tuple[typehints.Union[int, str], ...], lambda _list1, _list2: (*_list1, *_list2, *_list2), [typehints.List[int], typehints.List[str]])

    def testBuildSetUnpackOrUpdate(self):
        if False:
            return 10
        self.assertReturnType(typehints.Set[typehints.Union[int, str]], lambda _list1, _list2: {*_list1, *_list2, *_list2}, [typehints.List[int], typehints.List[str]])

    def testBuildMapUnpackOrUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(typehints.Dict[str, typehints.Union[int, str, float]], lambda a, b, c: {**a, **b, **c}, [typehints.Dict[str, int], typehints.Dict[str, str], typehints.List[typehints.Tuple[str, float]]])

    def testIdentity(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(int, lambda x: x, [int])

    def testIndexing(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(int, lambda x: x[0], [typehints.Tuple[int, str]])
        self.assertReturnType(str, lambda x: x[1], [typehints.Tuple[int, str]])
        self.assertReturnType(str, lambda x: x[1], [typehints.List[str]])

    def testTuples(self):
        if False:
            while True:
                i = 10
        self.assertReturnType(typehints.Tuple[typehints.Tuple[()], int], lambda x: ((), x), [int])
        self.assertReturnType(typehints.Tuple[str, int, float], lambda x: (x, 0, 1.0), [str])

    def testGetItem(self):
        if False:
            print('Hello World!')

        def reverse(ab):
            if False:
                while True:
                    i = 10
            return (ab[-1], ab[0])
        self.assertReturnType(typehints.Tuple[typehints.Any, typehints.Any], reverse, [typehints.Any])
        self.assertReturnType(typehints.Tuple[int, float], reverse, [typehints.Tuple[float, int]])
        self.assertReturnType(typehints.Tuple[int, str], reverse, [typehints.Tuple[str, float, int]])
        self.assertReturnType(typehints.Tuple[int, int], reverse, [typehints.List[int]])

    def testGetItemSlice(self):
        if False:
            print('Hello World!')
        self.assertReturnType(typehints.List[int], lambda v: v[::-1], [typehints.List[int]])
        self.assertReturnType(typehints.Tuple[int], lambda v: v[::-1], [typehints.Tuple[int]])
        self.assertReturnType(str, lambda v: v[::-1], [str])
        self.assertReturnType(typehints.Any, lambda v: v[::-1], [typehints.Any])
        self.assertReturnType(typehints.Any, lambda v: v[::-1], [object])
        test_list = ['a', 'b']
        self.assertReturnType(typehints.List[str], lambda : test_list[:], [])

    def testUnpack(self):
        if False:
            i = 10
            return i + 15

        def reverse(a_b):
            if False:
                for i in range(10):
                    print('nop')
            (a, b) = a_b
            return (b, a)
        any_tuple = typehints.Tuple[typehints.Any, typehints.Any]
        self.assertReturnType(typehints.Tuple[int, float], reverse, [typehints.Tuple[float, int]])
        self.assertReturnType(typehints.Tuple[int, int], reverse, [typehints.Tuple[int, ...]])
        self.assertReturnType(typehints.Tuple[int, int], reverse, [typehints.List[int]])
        self.assertReturnType(typehints.Tuple[typehints.Union[int, float, str], typehints.Union[int, float, str]], reverse, [typehints.Tuple[int, float, str]])
        self.assertReturnType(any_tuple, reverse, [typehints.Any])
        self.assertReturnType(typehints.Tuple[int, float], reverse, [trivial_inference.Const((1.0, 1))])
        self.assertReturnType(any_tuple, reverse, [trivial_inference.Const((1, 2, 3))])

    def testBuildMap(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(typehints.Dict[typehints.Any, typehints.Any], lambda k, v: {}, [int, float])
        self.assertReturnType(typehints.Dict[int, float], lambda k, v: {k: v}, [int, float])
        self.assertReturnType(typehints.Tuple[str, typehints.Dict[int, float]], lambda k, v: ('s', {k: v}), [int, float])
        self.assertReturnType(typehints.Dict[int, typehints.Union[float, str]], lambda k1, v1, k2, v2: {k1: v1, k2: v2}, [int, float, int, str])
        self.assertReturnType(typehints.Dict[str, typehints.Union[int, float]], lambda a, b: {'a': a, 'b': b}, [int, float])
        self.assertReturnType(typehints.Tuple[int, typehints.Dict[str, typehints.Union[int, float]]], lambda a, b: (4, {'a': a, 'b': b}), [int, float])

    def testNoneReturn(self):
        if False:
            while True:
                i = 10

        def func(a):
            if False:
                for i in range(10):
                    print('nop')
            if a == 5:
                return a
            return None
        self.assertReturnType(typehints.Union[int, type(None)], func, [int])

    def testSimpleList(self):
        if False:
            return 10
        self.assertReturnType(typehints.List[int], lambda xs: [1, 2], [typehints.Tuple[int, ...]])
        self.assertReturnType(typehints.List[typehints.Any], lambda xs: list(xs), [typehints.Tuple[int, ...]])

    def testListComprehension(self):
        if False:
            i = 10
            return i + 15
        self.assertReturnType(typehints.List[int], lambda xs: [x for x in xs], [typehints.Tuple[int, ...]])

    def testTupleListComprehension(self):
        if False:
            while True:
                i = 10
        self.assertReturnType(typehints.List[int], lambda xs: [x for x in xs], [typehints.Tuple[int, int, int]])
        self.assertReturnType(typehints.List[typehints.Union[int, float]], lambda xs: [x for x in xs], [typehints.Tuple[int, float]])
        expected = typehints.List[typehints.Tuple[str, int]]
        self.assertReturnType(expected, lambda kvs: [(kvs[0], v) for v in kvs[1]], [typehints.Tuple[str, typehints.Iterable[int]]])
        self.assertReturnType(typehints.List[typehints.Tuple[str, typehints.Union[str, int], int]], lambda L: [(a, a or b, b) for (a, b) in L], [typehints.Iterable[typehints.Tuple[str, int]]])

    def testGenerator(self):
        if False:
            while True:
                i = 10

        def foo(x, y):
            if False:
                print('Hello World!')
            yield x
            yield y
        self.assertReturnType(typehints.Iterable[int], foo, [int, int])
        self.assertReturnType(typehints.Iterable[typehints.Union[int, float]], foo, [int, float])

    def testGeneratorComprehension(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(typehints.Iterable[int], lambda xs: (x for x in xs), [typehints.Tuple[int, ...]])

    def testBinOp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(int, lambda a, b: a + b, [int, int])
        self.assertReturnType(int, lambda a: a + 1, [int])
        self.assertReturnType(typehints.Any, lambda a, b: a + b, [int, typehints.Any])
        self.assertReturnType(typehints.List[typehints.Union[int, str]], lambda a, b: a + b, [typehints.List[int], typehints.List[str]])

    def testCall(self):
        if False:
            return 10
        f = lambda x, *args: x
        self.assertReturnType(typehints.Tuple[int, float], lambda : (f(1), f(2.0, 3)))
        self.assertReturnType(typehints.Tuple[int, typehints.Any], lambda : (1, f(x=1.0)))

    def testCallNullaryMethod(self):
        if False:
            i = 10
            return i + 15

        class Foo:
            pass
        self.assertReturnType(typehints.Tuple[Foo, typehints.Any], lambda x: (x, x.unknown()), [Foo])

    def testCallNestedLambda(self):
        if False:
            return 10

        class Foo:
            pass
        self.assertReturnType(typehints.Tuple[Foo, int], lambda x: (x, (lambda : 3)()), [Foo])

    def testClosure(self):
        if False:
            while True:
                i = 10
        x = 1
        y = 1.0
        self.assertReturnType(typehints.Tuple[int, float], lambda : (x, y))

    @unittest.skip('https://github.com/apache/beam/issues/28420')
    def testLocalClosure(self):
        if False:
            while True:
                i = 10
        self.assertReturnType(typehints.Tuple[int, int], lambda x: (x, (lambda : x)()), [int])

    def testGlobals(self):
        if False:
            i = 10
            return i + 15
        self.assertReturnType(int, lambda : global_int)

    def testBuiltins(self):
        if False:
            return 10
        self.assertReturnType(int, lambda x: len(x), [typehints.Any])

    def testGetAttr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(typehints.Tuple[str, typehints.Any], lambda : (typehints.__doc__, typehints.fake))

    def testSetAttr(self):
        if False:
            i = 10
            return i + 15

        def fn(obj, flag):
            if False:
                while True:
                    i = 10
            if flag == 1:
                obj.attr = 1
                res = 1
            elif flag == 2:
                obj.attr = 2
                res = 1.5
            return res
        self.assertReturnType(typehints.Union[int, float], fn, [int])

    def testSetDeleteGlobal(self):
        if False:
            i = 10
            return i + 15

        def fn(flag):
            if False:
                for i in range(10):
                    print('nop')
            global global_var
            if flag == 1:
                global_var = 3
                res = 1
            elif flag == 4:
                del global_var
                res = 'str'
            return res
        self.assertReturnType(typehints.Union[int, str], fn, [int])

    def testMethod(self):
        if False:
            return 10

        class A(object):

            def m(self, x):
                if False:
                    return 10
                return x
        self.assertReturnType(int, lambda : A().m(3))
        self.assertReturnType(float, lambda : A.m(A(), 3.0))

    def testCallFunctionOnAny(self):
        if False:
            print('Hello World!')

        def call_function_on_any(s):
            if False:
                i = 10
                return i + 15
            s.split()
            return 0
        self.assertReturnType(int, call_function_on_any, [str])

    def testAlwaysReturnsEarly(self):
        if False:
            i = 10
            return i + 15

        def some_fn(v):
            if False:
                print('Hello World!')
            if v:
                return 1
            return 2
        self.assertReturnType(int, some_fn)

    def testDict(self):
        if False:
            print('Hello World!')
        self.assertReturnType(typehints.Dict[typehints.Any, typehints.Any], lambda : {})

    def testDictComprehension(self):
        if False:
            for i in range(10):
                print('nop')
        fields = []
        expected_type = typehints.Dict[typehints.Any, typehints.Any]
        self.assertReturnType(expected_type, lambda row: {f: row[f] for f in fields}, [typehints.Any])

    def testDictComprehensionSimple(self):
        if False:
            i = 10
            return i + 15
        self.assertReturnType(typehints.Dict[str, int], lambda _list: {'a': 1 for _ in _list}, [])

    def testSet(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(typehints.Set[typehints.Union[()]], lambda : {x for x in ()})
        self.assertReturnType(typehints.Set[int], lambda xs: {x for x in xs}, [typehints.List[int]])

    def testDepthFunction(self):
        if False:
            return 10

        def f(i):
            if False:
                for i in range(10):
                    print('nop')
            return i
        self.assertReturnType(typehints.Any, lambda i: f(i), [int], depth=0)
        self.assertReturnType(int, lambda i: f(i), [int], depth=1)

    def testDepthMethod(self):
        if False:
            for i in range(10):
                print('nop')

        class A(object):

            def m(self, x):
                if False:
                    while True:
                        i = 10
                return x
        self.assertReturnType(typehints.Any, lambda : A().m(3), depth=0)
        self.assertReturnType(int, lambda : A().m(3), depth=1)
        self.assertReturnType(typehints.Any, lambda : A.m(A(), 3.0), depth=0)
        self.assertReturnType(float, lambda : A.m(A(), 3.0), depth=1)

    def testBuildTupleUnpackWithCall(self):
        if False:
            print('Hello World!')

        def fn(x1, x2, *unused_args):
            if False:
                return 10
            return (x1, x2)
        self.assertReturnType(typehints.Tuple[typehints.Union[str, float, int], typehints.Union[str, float, int]], lambda x1, x2, _list: fn(x1, x2, *_list), [str, float, typehints.List[int]])
        self.assertReturnType(typehints.Tuple[typehints.Union[str, typehints.List[int]], typehints.Union[str, typehints.List[int]]], lambda x1, x2, _list: fn(x1, x2, *_list), [str, typehints.List[int]])

    def testCallFunctionEx(self):
        if False:
            while True:
                i = 10

        def fn(*args):
            if False:
                return 10
            return args
        self.assertReturnType(typehints.List[typehints.Union[str, float]], lambda x1, x2: fn(*[x1, x2]), [str, float])

    def testCallFunctionExKwargs(self):
        if False:
            while True:
                i = 10

        def fn(x1, x2, **unused_kwargs):
            if False:
                while True:
                    i = 10
            return (x1, x2)
        self.assertReturnType(typehints.Any, lambda x1, x2, _dict: fn(x1, x2, **_dict), [str, float, typehints.List[int]])

    def testInstanceToType(self):
        if False:
            i = 10
            return i + 15

        class MyClass(object):

            def method(self):
                if False:
                    while True:
                        i = 10
                pass
        test_cases = [(typehints.Dict[str, int], {'a': 1}), (typehints.Dict[str, typehints.Union[str, int]], {'a': 1, 'b': 'c'}), (typehints.Dict[typehints.Any, typehints.Any], {}), (typehints.Set[str], {'a'}), (typehints.Set[typehints.Union[str, float]], {'a', 0.4}), (typehints.Set[typehints.Any], set()), (typehints.FrozenSet[str], frozenset(['a'])), (typehints.FrozenSet[typehints.Union[str, float]], frozenset(['a', 0.4])), (typehints.FrozenSet[typehints.Any], frozenset()), (typehints.Tuple[int], (1,)), (typehints.Tuple[int, int, str], (1, 2, '3')), (typehints.Tuple[()], ()), (typehints.List[int], [1]), (typehints.List[typehints.Union[int, str]], [1, 'a']), (typehints.List[typehints.Any], []), (type(None), None), (type(MyClass), MyClass), (MyClass, MyClass()), (type(MyClass.method), MyClass.method), (types.MethodType, MyClass().method), (row_type.RowTypeConstraint.from_fields([('x', int)]), beam.Row(x=37))]
        for (expected_type, instance) in test_cases:
            self.assertEqual(expected_type, trivial_inference.instance_to_type(instance), msg=instance)

    def testRow(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReturnType(row_type.RowTypeConstraint.from_fields([('x', int), ('y', str)]), lambda x, y: beam.Row(x=x + 1, y=y), [int, str])
        self.assertReturnType(row_type.RowTypeConstraint.from_fields([('x', int), ('y', str)]), lambda x: beam.Row(x=x, y=str(x)), [int])

    def testRowAttr(self):
        if False:
            return 10
        self.assertReturnType(typehints.Tuple[int, str], lambda row: (row.x, getattr(row, 'y')), [row_type.RowTypeConstraint.from_fields([('x', int), ('y', str)])])

    def testPyCallable(self):
        if False:
            i = 10
            return i + 15
        self.assertReturnType(typehints.Tuple[int, str], python_callable.PythonCallableWithSource('lambda x: (x, str(x))'), [int])
if __name__ == '__main__':
    unittest.main()
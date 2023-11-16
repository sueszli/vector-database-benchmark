from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import EvaluationContext, GuardRejection, guarded_eval, _unbind_method
from IPython.testing import decorators as dec
import pytest

def create_context(evaluation: str, **kwargs):
    if False:
        while True:
            i = 10
    return EvaluationContext(locals=kwargs, globals={}, evaluation=evaluation)
forbidden = partial(create_context, 'forbidden')
minimal = partial(create_context, 'minimal')
limited = partial(create_context, 'limited')
unsafe = partial(create_context, 'unsafe')
dangerous = partial(create_context, 'dangerous')
LIMITED_OR_HIGHER = [limited, unsafe, dangerous]
MINIMAL_OR_HIGHER = [minimal, *LIMITED_OR_HIGHER]

@contextmanager
def module_not_installed(module: str):
    if False:
        i = 10
        return i + 15
    import sys
    try:
        to_restore = sys.modules[module]
        del sys.modules[module]
    except KeyError:
        to_restore = None
    try:
        yield
    finally:
        sys.modules[module] = to_restore

def test_external_not_installed():
    if False:
        return 10
    '\n    Because attribute check requires checking if object is not of allowed\n    external type, this tests logic for absence of external module.\n    '

    class Custom:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.test = 1

        def __getattr__(self, key):
            if False:
                return 10
            return key
    with module_not_installed('pandas'):
        context = limited(x=Custom())
        with pytest.raises(GuardRejection):
            guarded_eval('x.test', context)

@dec.skip_without('pandas')
def test_external_changed_api(monkeypatch):
    if False:
        i = 10
        return i + 15
    'Check that the execution rejects if external API changed paths'
    import pandas as pd
    series = pd.Series([1], index=['a'])
    with monkeypatch.context() as m:
        m.delattr(pd, 'Series')
        context = limited(data=series)
        with pytest.raises(GuardRejection):
            guarded_eval('data.iloc[0]', context)

@dec.skip_without('pandas')
def test_pandas_series_iloc():
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    series = pd.Series([1], index=['a'])
    context = limited(data=series)
    assert guarded_eval('data.iloc[0]', context) == 1

def test_rejects_custom_properties():
    if False:
        while True:
            i = 10

    class BadProperty:

        @property
        def iloc(self):
            if False:
                return 10
            return [None]
    series = BadProperty()
    context = limited(data=series)
    with pytest.raises(GuardRejection):
        guarded_eval('data.iloc[0]', context)

@dec.skip_without('pandas')
def test_accepts_non_overriden_properties():
    if False:
        i = 10
        return i + 15
    import pandas as pd

    class GoodProperty(pd.Series):
        pass
    series = GoodProperty([1], index=['a'])
    context = limited(data=series)
    assert guarded_eval('data.iloc[0]', context) == 1

@dec.skip_without('pandas')
def test_pandas_series():
    if False:
        return 10
    import pandas as pd
    context = limited(data=pd.Series([1], index=['a']))
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(KeyError):
        guarded_eval('data["c"]', context)

@dec.skip_without('pandas')
def test_pandas_bad_series():
    if False:
        return 10
    import pandas as pd

    class BadItemSeries(pd.Series):

        def __getitem__(self, key):
            if False:
                print('Hello World!')
            return 'CUSTOM_ITEM'

    class BadAttrSeries(pd.Series):

        def __getattr__(self, key):
            if False:
                return 10
            return 'CUSTOM_ATTR'
    bad_series = BadItemSeries([1], index=['a'])
    context = limited(data=bad_series)
    with pytest.raises(GuardRejection):
        guarded_eval('data["a"]', context)
    with pytest.raises(GuardRejection):
        guarded_eval('data["c"]', context)
    assert guarded_eval('data.a', context) == 'CUSTOM_ITEM'
    context = unsafe(data=bad_series)
    assert guarded_eval('data["a"]', context) == 'CUSTOM_ITEM'
    bad_attr_series = BadAttrSeries([1], index=['a'])
    context = limited(data=bad_attr_series)
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(GuardRejection):
        guarded_eval('data.a', context)

@dec.skip_without('pandas')
def test_pandas_dataframe_loc():
    if False:
        while True:
            i = 10
    import pandas as pd
    from pandas.testing import assert_series_equal
    data = pd.DataFrame([{'a': 1}])
    context = limited(data=data)
    assert_series_equal(guarded_eval('data.loc[:, "a"]', context), data['a'])

def test_named_tuple():
    if False:
        print('Hello World!')

    class GoodNamedTuple(NamedTuple):
        a: str
        pass

    class BadNamedTuple(NamedTuple):
        a: str

        def __getitem__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            return None
    good = GoodNamedTuple(a='x')
    bad = BadNamedTuple(a='x')
    context = limited(data=good)
    assert guarded_eval('data[0]', context) == 'x'
    context = limited(data=bad)
    with pytest.raises(GuardRejection):
        guarded_eval('data[0]', context)

def test_dict():
    if False:
        while True:
            i = 10
    context = limited(data={'a': 1, 'b': {'x': 2}, ('x', 'y'): 3})
    assert guarded_eval('data["a"]', context) == 1
    assert guarded_eval('data["b"]', context) == {'x': 2}
    assert guarded_eval('data["b"]["x"]', context) == 2
    assert guarded_eval('data["x", "y"]', context) == 3
    assert guarded_eval('data.keys', context)

def test_set():
    if False:
        return 10
    context = limited(data={'a', 'b'})
    assert guarded_eval('data.difference', context)

def test_list():
    if False:
        print('Hello World!')
    context = limited(data=[1, 2, 3])
    assert guarded_eval('data[1]', context) == 2
    assert guarded_eval('data.copy', context)

def test_dict_literal():
    if False:
        return 10
    context = limited()
    assert guarded_eval('{}', context) == {}
    assert guarded_eval('{"a": 1}', context) == {'a': 1}

def test_list_literal():
    if False:
        i = 10
        return i + 15
    context = limited()
    assert guarded_eval('[]', context) == []
    assert guarded_eval('[1, "a"]', context) == [1, 'a']

def test_set_literal():
    if False:
        print('Hello World!')
    context = limited()
    assert guarded_eval('set()', context) == set()
    assert guarded_eval('{"a"}', context) == {'a'}

def test_evaluates_if_expression():
    if False:
        return 10
    context = limited()
    assert guarded_eval('2 if True else 3', context) == 2
    assert guarded_eval('4 if False else 5', context) == 5

def test_object():
    if False:
        while True:
            i = 10
    obj = object()
    context = limited(obj=obj)
    assert guarded_eval('obj.__dir__', context) == obj.__dir__

@pytest.mark.parametrize('code,expected', [['int.numerator', int.numerator], ['float.is_integer', float.is_integer], ['complex.real', complex.real]])
def test_number_attributes(code, expected):
    if False:
        print('Hello World!')
    assert guarded_eval(code, limited()) == expected

def test_method_descriptor():
    if False:
        return 10
    context = limited()
    assert guarded_eval('list.copy.__name__', context) == 'copy'

class HeapType:
    pass

class CallCreatesHeapType:

    def __call__(self) -> HeapType:
        if False:
            for i in range(10):
                print('nop')
        return HeapType()

class CallCreatesBuiltin:

    def __call__(self) -> frozenset:
        if False:
            i = 10
            return i + 15
        return frozenset()

@pytest.mark.parametrize('data,good,bad,expected, equality', [[[1, 2, 3], 'data.index(2)', 'data.append(4)', 1, True], [{'a': 1}, 'data.keys().isdisjoint({})', 'data.update()', True, True], [CallCreatesHeapType(), 'data()', 'data.__class__()', HeapType, False], [CallCreatesBuiltin(), 'data()', 'data.__class__()', frozenset, False]])
def test_evaluates_calls(data, good, bad, expected, equality):
    if False:
        return 10
    context = limited(data=data)
    value = guarded_eval(good, context)
    if equality:
        assert value == expected
    else:
        assert isinstance(value, expected)
    with pytest.raises(GuardRejection):
        guarded_eval(bad, context)

@pytest.mark.parametrize('code,expected', [['(1\n+\n1)', 2], ['list(range(10))[-1:]', [9]], ['list(range(20))[3:-2:3]', [3, 6, 9, 12, 15]]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_complex_cases(code, expected, context):
    if False:
        return 10
    assert guarded_eval(code, context()) == expected

@pytest.mark.parametrize('code,expected', [['1', 1], ['1.0', 1.0], ['0xdeedbeef', 3740122863], ['True', True], ['None', None], ['{}', {}], ['[]', []]])
@pytest.mark.parametrize('context', MINIMAL_OR_HIGHER)
def test_evaluates_literals(code, expected, context):
    if False:
        for i in range(10):
            print('nop')
    assert guarded_eval(code, context()) == expected

@pytest.mark.parametrize('code,expected', [['-5', -5], ['+5', +5], ['~5', -6]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_unary_operations(code, expected, context):
    if False:
        while True:
            i = 10
    assert guarded_eval(code, context()) == expected

@pytest.mark.parametrize('code,expected', [['1 + 1', 2], ['3 - 1', 2], ['2 * 3', 6], ['5 // 2', 2], ['5 / 2', 2.5], ['5**2', 25], ['2 >> 1', 1], ['2 << 1', 4], ['1 | 2', 3], ['1 & 1', 1], ['1 & 2', 0]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_binary_operations(code, expected, context):
    if False:
        while True:
            i = 10
    assert guarded_eval(code, context()) == expected

@pytest.mark.parametrize('code,expected', [['2 > 1', True], ['2 < 1', False], ['2 <= 1', False], ['2 <= 2', True], ['1 >= 2', False], ['2 >= 2', True], ['2 == 2', True], ['1 == 2', False], ['1 != 2', True], ['1 != 1', False], ['1 < 4 < 3', False], ['(1 < 4) < 3', True], ['4 > 3 > 2 > 1', True], ['4 > 3 > 2 > 9', False], ['1 < 2 < 3 < 4', True], ['9 < 2 < 3 < 4', False], ['1 < 2 > 1 > 0 > -1 < 1', True], ['1 in [1] in [[1]]', True], ['1 in [1] in [[2]]', False], ['1 in [1]', True], ['0 in [1]', False], ['1 not in [1]', False], ['0 not in [1]', True], ['True is True', True], ['False is False', True], ['True is False', False], ['True is not True', False], ['False is not True', True]])
@pytest.mark.parametrize('context', LIMITED_OR_HIGHER)
def test_evaluates_comparisons(code, expected, context):
    if False:
        i = 10
        return i + 15
    assert guarded_eval(code, context()) == expected

def test_guards_comparisons():
    if False:
        print('Hello World!')

    class GoodEq(int):
        pass

    class BadEq(int):

        def __eq__(self, other):
            if False:
                return 10
            assert False
    context = limited(bad=BadEq(1), good=GoodEq(1))
    with pytest.raises(GuardRejection):
        guarded_eval('bad == 1', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad != 1', context)
    with pytest.raises(GuardRejection):
        guarded_eval('1 == bad', context)
    with pytest.raises(GuardRejection):
        guarded_eval('1 != bad', context)
    assert guarded_eval('good == 1', context) is True
    assert guarded_eval('good != 1', context) is False
    assert guarded_eval('1 == good', context) is True
    assert guarded_eval('1 != good', context) is False

def test_guards_unary_operations():
    if False:
        return 10

    class GoodOp(int):
        pass

    class BadOpInv(int):

        def __inv__(self, other):
            if False:
                i = 10
                return i + 15
            assert False

    class BadOpInverse(int):

        def __inv__(self, other):
            if False:
                i = 10
                return i + 15
            assert False
    context = limited(good=GoodOp(1), bad1=BadOpInv(1), bad2=BadOpInverse(1))
    with pytest.raises(GuardRejection):
        guarded_eval('~bad1', context)
    with pytest.raises(GuardRejection):
        guarded_eval('~bad2', context)

def test_guards_binary_operations():
    if False:
        i = 10
        return i + 15

    class GoodOp(int):
        pass

    class BadOp(int):

        def __add__(self, other):
            if False:
                print('Hello World!')
            assert False
    context = limited(good=GoodOp(1), bad=BadOp(1))
    with pytest.raises(GuardRejection):
        guarded_eval('1 + bad', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad + 1', context)
    assert guarded_eval('good + 1', context) == 2
    assert guarded_eval('1 + good', context) == 2

def test_guards_attributes():
    if False:
        print('Hello World!')

    class GoodAttr(float):
        pass

    class BadAttr1(float):

        def __getattr__(self, key):
            if False:
                i = 10
                return i + 15
            assert False

    class BadAttr2(float):

        def __getattribute__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            assert False
    context = limited(good=GoodAttr(0.5), bad1=BadAttr1(0.5), bad2=BadAttr2(0.5))
    with pytest.raises(GuardRejection):
        guarded_eval('bad1.as_integer_ratio', context)
    with pytest.raises(GuardRejection):
        guarded_eval('bad2.as_integer_ratio', context)
    assert guarded_eval('good.as_integer_ratio()', context) == (1, 2)

@pytest.mark.parametrize('context', MINIMAL_OR_HIGHER)
def test_access_builtins(context):
    if False:
        while True:
            i = 10
    assert guarded_eval('round', context()) == round

def test_access_builtins_fails():
    if False:
        return 10
    context = limited()
    with pytest.raises(NameError):
        guarded_eval('this_is_not_builtin', context)

def test_rejects_forbidden():
    if False:
        print('Hello World!')
    context = forbidden()
    with pytest.raises(GuardRejection):
        guarded_eval('1', context)

def test_guards_locals_and_globals():
    if False:
        print('Hello World!')
    context = EvaluationContext(locals={'local_a': 'a'}, globals={'global_b': 'b'}, evaluation='minimal')
    with pytest.raises(GuardRejection):
        guarded_eval('local_a', context)
    with pytest.raises(GuardRejection):
        guarded_eval('global_b', context)

def test_access_locals_and_globals():
    if False:
        return 10
    context = EvaluationContext(locals={'local_a': 'a'}, globals={'global_b': 'b'}, evaluation='limited')
    assert guarded_eval('local_a', context) == 'a'
    assert guarded_eval('global_b', context) == 'b'

@pytest.mark.parametrize('code', ['def func(): pass', 'class C: pass', 'x = 1', 'x += 1', 'del x', 'import ast'])
@pytest.mark.parametrize('context', [minimal(), limited(), unsafe()])
def test_rejects_side_effect_syntax(code, context):
    if False:
        print('Hello World!')
    with pytest.raises(SyntaxError):
        guarded_eval(code, context)

def test_subscript():
    if False:
        for i in range(10):
            print('nop')
    context = EvaluationContext(locals={}, globals={}, evaluation='limited', in_subscript=True)
    empty_slice = slice(None, None, None)
    assert guarded_eval('', context) == tuple()
    assert guarded_eval(':', context) == empty_slice
    assert guarded_eval('1:2:3', context) == slice(1, 2, 3)
    assert guarded_eval(':, "a"', context) == (empty_slice, 'a')

def test_unbind_method():
    if False:
        i = 10
        return i + 15

    class X(list):

        def index(self, k):
            if False:
                print('Hello World!')
            return 'CUSTOM'
    x = X()
    assert _unbind_method(x.index) is X.index
    assert _unbind_method([].index) is list.index
    assert _unbind_method(list.index) is None

def test_assumption_instance_attr_do_not_matter():
    if False:
        i = 10
        return i + 15
    "This is semi-specified in Python documentation.\n\n    However, since the specification says 'not guaranteed\n    to work' rather than 'is forbidden to work', future\n    versions could invalidate this assumptions. This test\n    is meant to catch such a change if it ever comes true.\n    "

    class T:

        def __getitem__(self, k):
            if False:
                i = 10
                return i + 15
            return 'a'

        def __getattr__(self, k):
            if False:
                i = 10
                return i + 15
            return 'a'

    def f(self):
        if False:
            while True:
                i = 10
        return 'b'
    t = T()
    t.__getitem__ = f
    t.__getattr__ = f
    assert t[1] == 'a'
    assert t[1] == 'a'

def test_assumption_named_tuples_share_getitem():
    if False:
        print('Hello World!')
    'Check assumption on named tuples sharing __getitem__'
    from typing import NamedTuple

    class A(NamedTuple):
        pass

    class B(NamedTuple):
        pass
    assert A.__getitem__ == B.__getitem__

@dec.skip_without('numpy')
def test_module_access():
    if False:
        return 10
    import numpy
    context = limited(numpy=numpy)
    assert guarded_eval('numpy.linalg.norm', context) == numpy.linalg.norm
    context = minimal(numpy=numpy)
    with pytest.raises(GuardRejection):
        guarded_eval('np.linalg.norm', context)
from builtins import _test_sink, _test_source
from typing import Dict, List, Tuple

def some_service(id):
    if False:
        i = 10
        return i + 15
    ...

def _unpack(tuple):
    if False:
        i = 10
        return i + 15
    ...

class DataRecord:
    ...

class Data:

    def __init__(self, a, b):
        if False:
            i = 10
            return i + 15
        self.a = a
        self.b = b

def get_data(x):
    if False:
        while True:
            i = 10
    return {'name': x, 'id': x}

def product_data(x):
    if False:
        i = 10
        return i + 15
    data = get_data(x)
    if x:
        parent = product_data(x.parent)
    else:
        parent = None
    is_blocked = some_service(data.id)
    report_tuple = DataRecord(id=data.id, username=data.name, isBlocked=is_blocked)
    return {'report': _unpack(report_tuple), 'id': data.id, 'parent_data': parent, 'name': data.name}

def product_data_wrapper(x):
    if False:
        print('Hello World!')
    return product_data(x)

def tito():
    if False:
        i = 10
        return i + 15
    return product_data_wrapper(_test_source())

def via_getattr(x, y):
    if False:
        while True:
            i = 10
    return getattr(x, 'foo', y)

class Recursive:

    def __init__(self, select):
        if False:
            for i in range(10):
                print('nop')
        self.init_list = [f[0].target.attname for f in select[self.cols_start:self.cols_end]]
        self.related = get_related(select)

def get_related(select):
    if False:
        for i in range(10):
            print('nop')
    return Recursive(select)

class FieldIsTITO:
    add_tito: int = 1

def adds_tito(x: FieldIsTITO) -> int:
    if False:
        for i in range(10):
            print('nop')
    return x.add_tito

class InheritsFromTITO(FieldIsTITO):
    pass

def adds_tito_inherited(x: InheritsFromTITO) -> int:
    if False:
        for i in range(10):
            print('nop')
    return x.add_tito

def adds_tito_with_indirect_sink(src: FieldIsTITO) -> None:
    if False:
        return 10
    indirect_sink(src)

def indirect_sink(x: FieldIsTITO) -> None:
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x.add_tito)

def issue_with_indirect_sink_tito():
    if False:
        i = 10
        return i + 15
    x = _test_source()
    adds_tito_with_indirect_sink(x)

def approximate_return_access_paths(x):
    if False:
        while True:
            i = 10
    return {'a': x.a, 'b': x.b, 'c': x.c, 'd': x.d, 'e': x.e, 'f': x.f, 'g': x.g, 'h': x.h, 'j': x.j, 'k': x.k, 'l': x.l}

async def return_taint(tainted: str, b1: str, b2: str) -> Tuple[str, str, str]:
    return (tainted, b1, b2)

async def test_tuple_tito_indices():
    (tainted, b1, b2) = await return_taint(_test_source(), '', '')
    _test_sink(b2)

def return_taint_in_list(tainted: str, a: str, b: str) -> List[str]:
    if False:
        print('Hello World!')
    return [tainted, a, b]

def add_feature(arg):
    if False:
        return 10
    return arg

def tito_with_feature(arg):
    if False:
        print('Hello World!')
    if arg:
        return arg
    else:
        return add_feature(arg)

def test_always_via_feature():
    if False:
        return 10
    _test_sink(tito_with_feature(_test_source()))

class GetQuery:

    def __init__(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.arg = arg

class GetUser(GetQuery):

    def __init__(self, arg):
        if False:
            while True:
                i = 10
        GetQuery.__init__(self, arg)

def test_explicit_call_to_superclass():
    if False:
        while True:
            i = 10
    user = GetUser(_test_source())
    _test_sink(user.arg)

def evaluate_lazy(payload: Dict[str, str]):
    if False:
        for i in range(10):
            print('nop')
    return {key: value for (key, value) in payload.items()}

def test_simplified_evaluator():
    if False:
        i = 10
        return i + 15
    _test_sink(evaluate_lazy(_test_source()))

class ComplexEvaluator:

    def evaluate_lazy_field(self, field):
        if False:
            while True:
                i = 10
        if callable(field):
            return field()
        else:
            return field

    def evaluate_lazy_payload(self, payload):
        if False:
            while True:
                i = 10

        def _evaluate(field):
            if False:
                i = 10
                return i + 15
            if isinstance(field, dict):
                return self.evaluate_lazy_payload(field)
            return self.evaluate_lazy_field(field)
        return {key: _evaluate(value) for (key, value) in payload.items()}

def test_complex_evaluator(evaluator: ComplexEvaluator):
    if False:
        while True:
            i = 10
    _test_sink(evaluator.evaluate_lazy_payload(_test_source()))

def obscure_tito(x):
    if False:
        print('Hello World!')
    ...

def into_dict_then_tito_collapse(x):
    if False:
        print('Hello World!')
    d = {'a': x}
    return obscure_tito(d)

def tito_collapse_then_into_dict(x):
    if False:
        for i in range(10):
            print('nop')
    y = obscure_tito(x)
    return {'a': y}

def issue_with_into_dict_then_tito_collapse():
    if False:
        while True:
            i = 10
    x = _test_source()
    y = into_dict_then_tito_collapse(x)
    _test_sink(y['b'])

def no_issue_with_tito_collapse_then_into_dict():
    if False:
        print('Hello World!')
    x = _test_source()
    y = tito_collapse_then_into_dict(x)
    _test_sink(y['b'])

def perfect_tito(x):
    if False:
        while True:
            i = 10
    return x

def into_dict_then_perfect_tito(x):
    if False:
        print('Hello World!')
    d = {'a': x}
    return perfect_tito(d)

def perfect_tito_then_into_dict(x):
    if False:
        i = 10
        return i + 15
    y = perfect_tito(x)
    return {'a': y}

def no_issue_with_into_dict_then_perfect_tito():
    if False:
        while True:
            i = 10
    x = _test_source()
    y = into_dict_then_perfect_tito(x)
    _test_sink(y['b'])

def no_issue_with_perfect_tito_then_into_dict():
    if False:
        for i in range(10):
            print('nop')
    x = _test_source()
    y = perfect_tito_then_into_dict(x)
    _test_sink(y['b'])

def issue_approximate_return_access_paths():
    if False:
        return 10
    x = object()
    x.a = _test_source()
    y = approximate_return_access_paths(x)
    _test_sink(y['a'])

def issue_approximate_return_access_paths_common_prefix():
    if False:
        i = 10
        return i + 15
    x = object()
    x.y.a = _test_source()
    y = approximate_return_access_paths(x)
    _test_sink(y['a'])

def non_issue_approximate_return_access_paths_common_prefix():
    if False:
        while True:
            i = 10
    x = object()
    x.a = _test_source()
    y = approximate_return_access_paths(x)
    _test_sink(y['a'])

def perfect_tito_with_tree_manipulation(x):
    if False:
        for i in range(10):
            print('nop')
    d = {'a': x}
    return d['a']

def tito_collapse_one_append_a_b_c(x):
    if False:
        for i in range(10):
            print('nop')
    return {'a': {'b': {'c': x}}}

def tito_collapse_one(x):
    if False:
        return 10
    y = tito_collapse_one_append_a_b_c(x)
    return y['a']['b']['c']

def tito_collapse_two_append_a_b(x):
    if False:
        i = 10
        return i + 15
    return {'a': {'b': x}}

def tito_collapse_two(x):
    if False:
        print('Hello World!')
    y = tito_collapse_two_append_a_b(x)
    return y['a']['b']

def tito_collapse_three_append_a(x):
    if False:
        print('Hello World!')
    return {'a': x}

def tito_collapse_three(x):
    if False:
        while True:
            i = 10
    y = tito_collapse_three_append_a(x)
    return y['a']

def into_dict_then_collapse_two(x):
    if False:
        for i in range(10):
            print('nop')
    d = {'a': x}
    return tito_collapse_two(d)

def collapse_two_then_into_dict(x):
    if False:
        while True:
            i = 10
    y = tito_collapse_two(x)
    return {'a': y}

def perfect_tito_then_into_deep_dict(x):
    if False:
        while True:
            i = 10
    y = perfect_tito(x)
    return {'a': {'b': {'c': {'d': {'e': y}}}}}

def collapse_two_then_into_deep_dict(x):
    if False:
        while True:
            i = 10
    y = tito_collapse_two(x)
    return {'a': {'b': {'c': {'d': {'e': y}}}}}

def combine_collapse_one(arg):
    if False:
        print('Hello World!')
    x = {'a': arg}
    y = tito_collapse_one(x)
    z = {'a': y}
    t = tito_collapse_one(z)
    return t

def combine_collapse_two(arg):
    if False:
        return 10
    x = {'a': arg}
    y = tito_collapse_two(x)
    z = {'a': y}
    t = tito_collapse_two(z)
    return t

def combine_collapse_three(arg):
    if False:
        for i in range(10):
            print('nop')
    x = {'a': arg}
    y = tito_collapse_three(x)
    z = {'a': y}
    t = tito_collapse_three(z)
    return t

def combine_collapse_two_and_one(arg):
    if False:
        return 10
    x = {'a': arg}
    y = tito_collapse_two(x)
    z = {'a': y}
    t = tito_collapse_one(z)
    return t

def combine_collapse_one_and_two(arg):
    if False:
        return 10
    x = {'a': arg}
    y = tito_collapse_one(x)
    z = {'a': y}
    t = tito_collapse_two(z)
    return t

def loop_perfect_tito(x):
    if False:
        while True:
            i = 10
    for _ in range(100):
        x = {'a': x}
        x = perfect_tito(x)
    return x

def loop_tito_collapse_one(x):
    if False:
        return 10
    for _ in range(100):
        x = {'a': x}
        x = tito_collapse_one(x)
    return x

def loop_tito_collapse_two(x):
    if False:
        for i in range(10):
            print('nop')
    for _ in range(100):
        x = {'a': x}
        x = tito_collapse_two(x)
    return x

def join_tito_collapse_test_1(x):
    if False:
        print('Hello World!')
    result = object()
    if 1 > 2:
        result.a = tito_collapse_two(x)
    else:
        result.a.b = tito_collapse_one(x)
    return result

def join_tito_collapse_test_2(x):
    if False:
        while True:
            i = 10
    result = object()
    if 1 > 2:
        result.a = tito_collapse_two(x)
    else:
        result.a.b = tito_collapse_three(x)
    return result

def tito_collapse_one_with_input_path(x):
    if False:
        return 10
    return tito_collapse_one(x['a']['b'])

def tito_collapse_one_with_input_path_with_hop(x):
    if False:
        print('Hello World!')
    return tito_collapse_one_with_input_path(x)

def no_issue_tito_collapse_two_with_input_path():
    if False:
        for i in range(10):
            print('nop')
    x = {'a': {'b': {'c': _test_source(), 'd': 0}}}
    y = tito_collapse_one_with_input_path(x)
    _test_sink(y['d'])

def join_tito_collapse_test_3(x):
    if False:
        print('Hello World!')
    if 1 > 2:
        return tito_collapse_one(x)
    else:
        return {'foo': tito_collapse_two(x)}

def issue_join_tito_collapse_test_3():
    if False:
        while True:
            i = 10
    x = {'a': _test_source()}
    y = join_tito_collapse_test_3(x)
    _test_sink(y['foo']['a'])

def user_declared_tito_no_collapse(arg):
    if False:
        return 10
    return

def no_issue_user_declared_tito_no_collapse():
    if False:
        for i in range(10):
            print('nop')
    x = {'a': _test_source()}
    y = user_declared_tito_no_collapse(x)
    _test_sink(y['b'])

def user_declared_tito_collapse_one(arg):
    if False:
        i = 10
        return i + 15
    return

def no_issue_user_declared_tito_collapse_one():
    if False:
        return 10
    x = {'a': _test_source()}
    y = user_declared_tito_collapse_one(x)
    _test_sink(y['b'])

def issue_user_declared_tito_collapse_one():
    if False:
        while True:
            i = 10
    x = {'a': {'b': _test_source()}}
    y = user_declared_tito_collapse_one(x)
    _test_sink(y['a']['c'])

def no_tito_init_then_overwrite(x):
    if False:
        print('Hello World!')
    d = {'a': x}
    d['a'] = 0
    return d

def no_tito_overwrite_then_init(d):
    if False:
        return 10
    d['a'] = 0
    return d['a']

def tito_with_sink(d):
    if False:
        for i in range(10):
            print('nop')
    x = d['a']
    _test_sink(d['a'])
    return x
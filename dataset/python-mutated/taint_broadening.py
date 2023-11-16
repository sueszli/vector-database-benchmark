from builtins import _test_sink, _test_source
from typing import Dict, Optional, Union, TypeVar

def tito(x):
    if False:
        for i in range(10):
            print('nop')
    ...

def sink_a(parameter):
    if False:
        i = 10
        return i + 15
    _test_sink(parameter['a'])

def tito_collapse_issue():
    if False:
        while True:
            i = 10
    a = {'a': _test_source(), 'b': 'b'}
    b = tito(a)
    _test_sink(b['b'])

def tito_collapse_sink(parameter):
    if False:
        while True:
            i = 10
    b = tito(parameter)
    _test_sink(b['b'])

def tito_collapse_source():
    if False:
        print('Hello World!')
    a = {'a': _test_source(), 'b': 'b'}
    return tito(a)

def issue_collapse():
    if False:
        while True:
            i = 10
    a = {'a': _test_source(), 'b': 'b'}
    _test_sink(a)

def model_broadening_collapse_source_width(c):
    if False:
        return 10
    result = {}
    if c:
        result['o1.1'] = _test_source()
        result['o1.2'] = _test_source()
        result['o1.3'] = _test_source()
        result['o1.4'] = _test_source()
        result['o1.5'] = _test_source()
        result['o1.6'] = _test_source()
        result['o1.7'] = _test_source()
        result['o1.8'] = _test_source()
        result['o1.9'] = _test_source()
        result['o1.10'] = _test_source()
        result['o1.11'] = _test_source()
        result['o1.12'] = _test_source()
        result['o1.13'] = _test_source()
        result['o1.14'] = _test_source()
        result['o1.15'] = _test_source()
    else:
        result['o2.1'] = _test_source()
        result['o2.2'] = _test_source()
        result['o2.3'] = _test_source()
        result['o2.4'] = _test_source()
        result['o2.5'] = _test_source()
        result['o2.6'] = _test_source()
        result['o2.7'] = _test_source()
        result['o2.8'] = _test_source()
        result['o2.9'] = _test_source()
        result['o2.10'] = _test_source()
        result['o2.11'] = _test_source()
        result['o2.12'] = _test_source()
        result['o2.13'] = _test_source()
        result['o2.14'] = _test_source()
        result['o2.15'] = _test_source()
    return result

def model_broadening_collapse_sink_width(parameter, condition):
    if False:
        i = 10
        return i + 15
    if condition:
        _test_sink(parameter['i1.1'])
        _test_sink(parameter['i1.2'])
        _test_sink(parameter['i1.3'])
        _test_sink(parameter['i1.4'])
        _test_sink(parameter['i1.5'])
        _test_sink(parameter['i1.6'])
        _test_sink(parameter['i1.7'])
        _test_sink(parameter['i1.8'])
        _test_sink(parameter['i1.9'])
        _test_sink(parameter['i1.10'])
        _test_sink(parameter['i1.11'])
        _test_sink(parameter['i1.12'])
        _test_sink(parameter['i1.13'])
        _test_sink(parameter['i1.14'])
        _test_sink(parameter['i1.15'])
    else:
        _test_sink(parameter['i2.1'])
        _test_sink(parameter['i2.2'])
        _test_sink(parameter['i2.3'])
        _test_sink(parameter['i2.4'])
        _test_sink(parameter['i2.5'])
        _test_sink(parameter['i2.6'])
        _test_sink(parameter['i2.7'])
        _test_sink(parameter['i2.8'])
        _test_sink(parameter['i2.9'])
        _test_sink(parameter['i2.10'])
        _test_sink(parameter['i2.11'])
        _test_sink(parameter['i2.12'])
        _test_sink(parameter['i2.13'])
        _test_sink(parameter['i2.14'])
        _test_sink(parameter['i2.15'])

def model_broadening_source_no_collapse_depth(condition):
    if False:
        for i in range(10):
            print('nop')
    result = {}
    if condition:
        result['a']['a']['a']['a']['1'] = _test_source()
    else:
        result['a']['a']['a']['a']['2'] = _test_source()
    return result

def source_taint_widening_collapse_depth():
    if False:
        while True:
            i = 10
    result = {}
    for _ in range(1000000):
        result = {'a': result, 'b': _test_source()}
    return result

def model_broadening_sink_no_collapse_depth(condition, parameter):
    if False:
        i = 10
        return i + 15
    if condition:
        _test_sink(parameter['a']['a']['a']['a']['1'])
    else:
        _test_sink(parameter['a']['a']['a']['a']['2'])

def sink_taint_widening_collapse_depth(parameter):
    if False:
        while True:
            i = 10
    for _ in range(1000000):
        _test_sink(parameter['b'])
        parameter = parameter['a']

def recursive_sink_parent(obj):
    if False:
        print('Hello World!')
    if obj.parent is not None:
        recursive_sink_parent(obj.parent)
    else:
        _test_sink(obj)

def recursive_sink_parent_attribute(obj):
    if False:
        print('Hello World!')
    if obj.parent is not None:
        recursive_sink_parent_attribute(obj.parent)
    else:
        _test_sink(obj.attribute)

def tito_broaden_input_and_output_paths(parameter) -> Dict[str, Union[str, Optional[int]]]:
    if False:
        while True:
            i = 10
    result: Dict[str, Union[str, Optional[int]]] = {}
    result['o1'] = parameter.i1
    result['o2'] = parameter.i2
    result['o3'] = parameter.i3
    result['o4'] = parameter.i4
    result['o5'] = parameter.i5
    result['o6'] = parameter.i6
    result['o7'] = parameter.i7
    result['o8'] = parameter.i8
    result['o9'] = parameter.i9
    result['o10'] = parameter.i10
    result['o11'] = parameter.i11
    return result

def tito_broaden_input_paths_but_not_output_path(parameter) -> Dict[str, Union[str, Optional[int]]]:
    if False:
        i = 10
        return i + 15
    result: Dict[str, Union[str, Optional[int]]] = {}
    result['o1'] = parameter.i1
    result['o2'] = parameter.i2
    result['o3'] = parameter.i3
    result['o4'] = parameter.i4
    result['o5'] = parameter.i5
    result['o6'] = parameter.i6
    result['o7'] = parameter.i7
    result['o8'] = parameter.i8
    result['o9'] = parameter.i9
    result['o10'] = parameter.i10
    return result

def random_tito(parameter, condition):
    if False:
        while True:
            i = 10
    if condition == 0:
        return parameter.i1
    elif condition == 1:
        return parameter.i2
    elif condition == 2:
        return parameter.i3
    else:
        return parameter.i4

def tito_broaden_output_paths_but_not_input_path(parameter, condition) -> Dict[str, Union[str, Optional[int]]]:
    if False:
        while True:
            i = 10
    result: Dict[str, Union[str, Optional[int]]] = {}
    result['o1'] = random_tito(parameter, condition)
    result['o2'] = random_tito(parameter, condition)
    result['o3'] = random_tito(parameter, condition)
    result['o4'] = random_tito(parameter, condition)
    result['o5'] = random_tito(parameter, condition)
    result['o6'] = random_tito(parameter, condition)
    result['o7'] = random_tito(parameter, condition)
    result['o8'] = random_tito(parameter, condition)
    result['o9'] = random_tito(parameter, condition)
    result['o10'] = random_tito(parameter, condition)
    result['o11'] = random_tito(parameter, condition)
    result['o12'] = random_tito(parameter, condition)
    result['o13'] = random_tito(parameter, condition)
    result['o14'] = random_tito(parameter, condition)
    result['o15'] = random_tito(parameter, condition)
    return result

def test_different_tito_broadenings():
    if False:
        i = 10
        return i + 15
    source = _test_source()
    kvs = tito_broaden_input_and_output_paths(source)
    _test_sink(f"\n            {', '.join(kvs.keys())}  # False positive here\n        ")
    kvs2 = tito_broaden_input_paths_but_not_output_path(source)
    _test_sink(f"\n            {', '.join(kvs2.keys())}  # No issue here\n        ")

def tito_broaden_input_and_output_paths_single_statement(x):
    if False:
        print('Hello World!')
    return {'a': x.a, 'b': x.b, 'c': x.c, 'd': x.d, 'e': x.e, 'f': x.f, 'g': x.g, 'h': x.h, 'j': x.j, 'k': x.k, 'l': x.l}

def tito_broaden_input_path_common_prefix(x):
    if False:
        return 10
    return {'a': x.y.a, 'b': x.y.b, 'c': x.y.c, 'd': x.y.d, 'e': x.y.e, 'f': x.y.f, 'g': x.y.g, 'h': x.y.h, 'j': x.y.j, 'k': x.y.k, 'l': x.y.l}

def tito_broaden_output_path_common_prefix(x):
    if False:
        return 10
    return {'a': {'a': x.a, 'b': x.b, 'c': x.c, 'd': x.d, 'e': x.e, 'f': x.f, 'g': x.g, 'h': x.h, 'j': x.j, 'k': x.k, 'l': x.l}}
T = TypeVar('T')

def skip_model_broadening(f: T) -> T:
    if False:
        return 10
    return f

@skip_model_broadening
def model_broadening_no_collapse_source_width(c):
    if False:
        print('Hello World!')
    result = {}
    if c:
        result['o1.1'] = _test_source()
        result['o1.2'] = _test_source()
        result['o1.3'] = _test_source()
        result['o1.4'] = _test_source()
        result['o1.5'] = _test_source()
        result['o1.6'] = _test_source()
        result['o1.7'] = _test_source()
        result['o1.8'] = _test_source()
        result['o1.9'] = _test_source()
        result['o1.10'] = _test_source()
        result['o1.11'] = _test_source()
        result['o1.12'] = _test_source()
        result['o1.13'] = _test_source()
        result['o1.14'] = _test_source()
        result['o1.15'] = _test_source()
    else:
        result['o2.1'] = _test_source()
        result['o2.2'] = _test_source()
        result['o2.3'] = _test_source()
        result['o2.4'] = _test_source()
        result['o2.5'] = _test_source()
        result['o2.6'] = _test_source()
        result['o2.7'] = _test_source()
        result['o2.8'] = _test_source()
        result['o2.9'] = _test_source()
        result['o2.10'] = _test_source()
        result['o2.11'] = _test_source()
        result['o2.12'] = _test_source()
        result['o2.13'] = _test_source()
        result['o2.14'] = _test_source()
        result['o2.15'] = _test_source()
    return result

@skip_model_broadening
def model_broadening_no_collapse_sink_width(parameter, condition):
    if False:
        while True:
            i = 10
    if condition:
        _test_sink(parameter['i1.1'])
        _test_sink(parameter['i1.2'])
        _test_sink(parameter['i1.3'])
        _test_sink(parameter['i1.4'])
        _test_sink(parameter['i1.5'])
        _test_sink(parameter['i1.6'])
        _test_sink(parameter['i1.7'])
        _test_sink(parameter['i1.8'])
        _test_sink(parameter['i1.9'])
        _test_sink(parameter['i1.10'])
        _test_sink(parameter['i1.11'])
        _test_sink(parameter['i1.12'])
        _test_sink(parameter['i1.13'])
        _test_sink(parameter['i1.14'])
        _test_sink(parameter['i1.15'])
    else:
        _test_sink(parameter['i2.1'])
        _test_sink(parameter['i2.2'])
        _test_sink(parameter['i2.3'])
        _test_sink(parameter['i2.4'])
        _test_sink(parameter['i2.5'])
        _test_sink(parameter['i2.6'])
        _test_sink(parameter['i2.7'])
        _test_sink(parameter['i2.8'])
        _test_sink(parameter['i2.9'])
        _test_sink(parameter['i2.10'])
        _test_sink(parameter['i2.11'])
        _test_sink(parameter['i2.12'])
        _test_sink(parameter['i2.13'])
        _test_sink(parameter['i2.14'])
        _test_sink(parameter['i2.15'])

@skip_model_broadening
def tito_no_broadening_input_and_output_paths(parameter) -> Dict[str, Union[str, Optional[int]]]:
    if False:
        while True:
            i = 10
    result: Dict[str, Union[str, Optional[int]]] = {}
    result['o1'] = parameter.i1
    result['o2'] = parameter.i2
    result['o3'] = parameter.i3
    result['o4'] = parameter.i4
    result['o5'] = parameter.i5
    result['o6'] = parameter.i6
    result['o7'] = parameter.i7
    result['o8'] = parameter.i8
    result['o9'] = parameter.i9
    result['o10'] = parameter.i10
    result['o11'] = parameter.i11
    return result
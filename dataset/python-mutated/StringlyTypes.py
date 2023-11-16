from typing import TypedDict
TypedDict.robot_not_keyword = True

class StringiFied(TypedDict):
    simple: 'int'
    params: 'List[Integer]'
    union: 'int | float'

def parameterized_list(argument: 'list[int]', expected=None):
    if False:
        return 10
    assert argument == eval(expected), repr(argument)

def parameterized_dict(argument: 'dict[int, float]', expected=None):
    if False:
        while True:
            i = 10
    assert argument == eval(expected), repr(argument)

def parameterized_set(argument: 'set[float]', expected=None):
    if False:
        for i in range(10):
            print('nop')
    assert argument == eval(expected), repr(argument)

def parameterized_tuple(argument: 'tuple[int,float,     str   ]', expected=None):
    if False:
        print('Hello World!')
    assert argument == eval(expected), repr(argument)

def homogenous_tuple(argument: 'tuple[int, ...]', expected=None):
    if False:
        return 10
    assert argument == eval(expected), repr(argument)

def union(argument: 'int | float', expected=None):
    if False:
        return 10
    assert argument == eval(expected), repr(argument)

def nested(argument: 'dict[int|float, tuple[int, ...] | tuple[int, float]]', expected=None):
    if False:
        return 10
    assert argument == eval(expected), repr(argument)

def aliases(a: 'sequence[integer]', b: 'MAPPING[STRING, DOUBLE|None]'):
    if False:
        return 10
    assert a == [1, 2, 3]
    assert b == {'1': 1.1, '2': 2.2, '': None}

def typeddict(argument: StringiFied):
    if False:
        print('Hello World!')
    assert argument['simple'] == 42
    assert argument['params'] == [1, 2, 3]
    assert argument['union'] == 3.14

def invalid(argument: 'bad[info'):
    if False:
        print('Hello World!')
    assert False

def bad_params(argument: 'list[int, str]'):
    if False:
        i = 10
        return i + 15
    assert False
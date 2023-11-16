from builtins import _test_source
from typing import Any, Type, Union
from django.http import Request
request: Request = ...

class StrIsTainted:

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return request.GET['tainted']

class ReprIsTainted:

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return request.GET['tainted']

def str_is_tainted():
    if False:
        while True:
            i = 10
    s = StrIsTainted()
    eval(str(s))

def repr_is_tainted():
    if False:
        i = 10
        return i + 15
    r = ReprIsTainted()
    eval(repr(r))

def str_falls_back_to_repr():
    if False:
        i = 10
        return i + 15
    r = ReprIsTainted()
    eval(str(r))

def implicit_str():
    if False:
        for i in range(10):
            print('nop')
    s = StrIsTainted()
    eval(f'prefix{s}suffix')

def implicit_repr():
    if False:
        i = 10
        return i + 15
    r = ReprIsTainted()
    eval(f'prefix{r}suffix')

def explicit_str():
    if False:
        while True:
            i = 10
    s = StrIsTainted()
    eval(f'prefix{s.__str__()}suffix')

class A:

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __str__(self):
        if False:
            return 10
        return self.value

def propagate_taint():
    if False:
        print('Hello World!')
    eval(f"{A(request.GET['tainted'])}")

def not_propagate_taint():
    if False:
        while True:
            i = 10
    eval(f"{A('not tainted')}")

def multiple_targets_for_single_expression(x: Union[A, StrIsTainted]):
    if False:
        print('Hello World!')
    eval(f'{x}')

class B:
    f: str = ''

    def __str__(self):
        if False:
            print('Hello World!')
        return self.f

class C:
    g: str = ''

    def __str__(self):
        if False:
            return 10
        return self.g

def join_source_and_attribute_source(i: int):
    if False:
        print('Hello World!')
    if i > 0:
        a: str = request.GET['tainted']
    else:
        a: C = C()
    eval(f'{a}')

def multiple_targets_for_single_expression_2(a: Union[int, B, C]):
    if False:
        while True:
            i = 10
    eval(f'{a}')

def joined_base():
    if False:
        return 10
    a = request.GET['tainted']
    b = 'benign'
    eval(f'{a}{b}')

def analyze_implicit_call():
    if False:
        while True:
            i = 10
    b = B()
    b.f = request.GET['tainted']
    eval(f'{str(b)}')
    eval(f'{b}')

def multiple_targets_for_single_expression_3(b_or_c: Union[B, C], d: int):
    if False:
        return 10
    a = 1
    return f'{a}{b_or_c}{d}'

def tito_f(x):
    if False:
        print('Hello World!')
    return x

def tito_g(y):
    if False:
        i = 10
        return i + 15
    return y

def compute_tito(x, y):
    if False:
        return 10
    return f'{tito_g(y)}{tito_f(x)}'

class D:

    def __str__(self):
        if False:
            print('Hello World!')
        return 'benign'

def forward_unioned_callees():
    if False:
        print('Hello World!')
    x: Union[str, D] = _test_source()
    return f'{x}'

def forward_unioned_callees_2():
    if False:
        return 10
    x: Union[Any, D] = _test_source()
    return f'{x}'

def backward_unioned_callees(x: Union[str, D]):
    if False:
        print('Hello World!')
    return f'{x}'

def backward_unioned_callees_2(x: Union[Any, D]):
    if False:
        return 10
    return f'{x}'

def any_type(x: Any):
    if False:
        return 10
    return f'{x}'

def object_type(x: object):
    if False:
        i = 10
        return i + 15
    return f'{x}'

class OverrideStr(float):

    def __str__(self):
        if False:
            return 10
        x = _test_source()
        return f'{x}'

def base_exception(e: Exception):
    if False:
        i = 10
        return i + 15
    return f'{type(e)}'

def function_call_target_1(error_type: Union[str, Type[Exception]]):
    if False:
        while True:
            i = 10
    f'{error_type}'

def function_call_target_2(x: Union[B, C]):
    if False:
        for i in range(10):
            print('nop')
    f'{x.__class__}'

def multiple_callees_same_location():
    if False:
        print('Hello World!')
    s = StrIsTainted()
    return str(s) + 'hello'
def get_name():
    if False:
        i = 10
        return i + 15
    return self.name

def get_name():
    if False:
        return 10
    return (self.name,)

def get_name():
    if False:
        print('Hello World!')
    del self.name

def get_name(self):
    if False:
        i = 10
        return i + 15
    return self.name
x = list()

def randdec(maxprec, maxexp):
    if False:
        while True:
            i = 10
    return numeric_string(maxprec, maxexp)

def ternary_optarg(prec, exp_range, itr):
    if False:
        for i in range(10):
            print('nop')
    for _ in range(100):
        a = randdec(prec, 2 * exp_range)
        b = randdec(prec, 2 * exp_range)
        c = randdec(prec, 2 * exp_range)
        yield (a, b, c, None)
        yield (a, b, c, None, None)

class Foo:
    CLASS_VAR = 1
    REFERENCES_CLASS_VAR = {'CLASS_VAR': CLASS_VAR}
    ANNOTATED_CLASS_VAR: int = 2
from typing import Literal

class Class:
    copy_on_model_validation: Literal['none', 'deep', 'shallow']
    post_init_call: Literal['before_validation', 'after_validation']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        Class
try:
    x = 1 / 0
except Exception as e:
    print(e)
y: int = 1
x: 'Bar' = 1
[first] = ['yup']
from typing import List, TypedDict

class Item(TypedDict):
    nodes: List[TypedDict('Node', {'name': str})]
from enum import Enum

class Ticket:

    class Status(Enum):
        OPEN = 'OPEN'
        CLOSED = 'CLOSED'

    def set_status(self, status: Status):
        if False:
            return 10
        self.status = status

def update_tomato():
    if False:
        for i in range(10):
            print('nop')
    print(TOMATO)
    TOMATO = 'cherry tomato'
A = f'{B}'
A = f'B{B}'
C = f'{A:{B}}'
C = f"{A:{f'{B}'}}"
from typing import Annotated, Literal

def arbitrary_callable() -> None:
    if False:
        print('Hello World!')
    ...

class PEP593Test:
    field: Annotated[int, 'base64', arbitrary_callable(), 123, (1, 2, 3)]
    field_with_stringified_type: Annotated['PEP593Test', 123]
    field_with_undefined_stringified_type: Annotated['PEP593Test123', 123]
    field_with_nested_subscript: Annotated[dict[Literal['foo'], str], 123]
    field_with_undefined_nested_subscript: Annotated[dict['foo', 'bar'], 123]

def in_ipython_notebook() -> bool:
    if False:
        while True:
            i = 10
    try:
        get_ipython()
    except NameError:
        return False
    return True

def named_expr():
    if False:
        for i in range(10):
            print('nop')
    if any(((key := (value := x)) for x in ['ok'])):
        print(key)
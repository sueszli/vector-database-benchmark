from __future__ import annotations
from typing import Annotated, Callable, List, Literal, NamedTuple, Tuple, TypeVar, TypedDict, cast
from mypy_extensions import Arg, DefaultArg, DefaultNamedArg, NamedArg, VarArg

def foo(var: 'MyClass') -> 'MyClass':
    if False:
        while True:
            i = 10
    x: 'MyClass'

def foo(*, inplace: 'bool'):
    if False:
        return 10
    pass

def foo(*args: 'str', **kwargs: 'int'):
    if False:
        for i in range(10):
            print('nop')
    pass
x: Tuple['MyClass']
x: Callable[['MyClass'], None]

class Foo(NamedTuple):
    x: 'MyClass'

class D(TypedDict):
    E: TypedDict('E', foo='int', total=False)

class D(TypedDict):
    E: TypedDict('E', {'foo': 'int'})
x: Annotated['str', 'metadata']
x: Arg('str', 'name')
x: DefaultArg('str', 'name')
x: NamedArg('str', 'name')
x: DefaultNamedArg('str', 'name')
x: DefaultNamedArg('str', name='name')
x: VarArg('str')
x: List[List[List['MyClass']]]
x: NamedTuple('X', [('foo', 'int'), ('bar', 'str')])
x: NamedTuple('X', fields=[('foo', 'int'), ('bar', 'str')])
x: NamedTuple(typename='X', fields=[('foo', 'int')])
X: MyCallable('X')

class D(TypedDict):
    E: TypedDict('E')
x: Annotated[()]
x: DefaultNamedArg(name='name', quox='str')
x: DefaultNamedArg(name='name')
x: NamedTuple('X', [('foo',), ('bar',)])
x: NamedTuple('X', ['foo', 'bar'])
x: NamedTuple()
x: Literal['foo', 'bar']
x = cast(x, 'str')

def foo(x, *args, **kwargs):
    if False:
        while True:
            i = 10
    ...

def foo(*, inplace):
    if False:
        i = 10
        return i + 15
    ...
x: Annotated[1:2] = ...
x = TypeVar('x', 'str', 'int')
x = cast('str', x)
X = List['MyClass']
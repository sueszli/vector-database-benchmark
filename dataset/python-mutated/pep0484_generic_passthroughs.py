from typing import Any, Callable, Iterable, List, Sequence, Tuple, Type, TypeVar, Union, Generic
T = TypeVar('T')
U = TypeVar('U')
TList = TypeVar('TList', bound=List[Any])
TType = TypeVar('TType', bound=Type)
TTypeAny = TypeVar('TTypeAny', bound=Type[Any])
TCallable = TypeVar('TCallable', bound=Callable[..., Any])
untyped_list_str = ['abc', 'def']
typed_list_str: List[str] = ['abc', 'def']
untyped_tuple_str = ('abc',)
typed_tuple_str: Tuple[str] = ('abc',)
untyped_tuple_str_int = ('abc', 4)
typed_tuple_str_int: Tuple[str, int] = ('abc', 4)
variadic_tuple_str: Tuple[str, ...] = ('abc',)
variadic_tuple_str_int: Tuple[Union[str, int], ...] = ('abc', 4)

def untyped_passthrough(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def typed_list_generic_passthrough(x: List[T]) -> List[T]:
    if False:
        print('Hello World!')
    return x

def typed_tuple_generic_passthrough(x: Tuple[T]) -> Tuple[T]:
    if False:
        print('Hello World!')
    return x

def typed_multi_typed_tuple_generic_passthrough(x: Tuple[T, U]) -> Tuple[U, T]:
    if False:
        while True:
            i = 10
    return (x[1], x[0])

def typed_variadic_tuple_generic_passthrough(x: Tuple[T, ...]) -> Sequence[T]:
    if False:
        print('Hello World!')
    return x

def typed_iterable_generic_passthrough(x: Iterable[T]) -> Iterable[T]:
    if False:
        print('Hello World!')
    return x

def typed_fully_generic_passthrough(x: T) -> T:
    if False:
        while True:
            i = 10
    return x

def typed_bound_generic_passthrough(x: TList) -> TList:
    if False:
        i = 10
        return i + 15
    x
    return x

def typed_quoted_return_generic_passthrough(x: T) -> 'List[T]':
    if False:
        i = 10
        return i + 15
    return [x]

def typed_quoted_input_generic_passthrough(x: 'Tuple[T]') -> T:
    if False:
        for i in range(10):
            print('nop')
    x
    return x[0]
for a in untyped_passthrough(untyped_list_str):
    a
for b in untyped_passthrough(typed_list_str):
    b
for c in typed_list_generic_passthrough(untyped_list_str):
    c
for d in typed_list_generic_passthrough(typed_list_str):
    d
for e in typed_iterable_generic_passthrough(untyped_list_str):
    e
for f in typed_iterable_generic_passthrough(typed_list_str):
    f
for g in typed_tuple_generic_passthrough(untyped_tuple_str):
    g
for h in typed_tuple_generic_passthrough(typed_tuple_str):
    h
out_untyped = typed_multi_typed_tuple_generic_passthrough(untyped_tuple_str_int)
out_untyped[0]
out_untyped[1]
out_typed = typed_multi_typed_tuple_generic_passthrough(typed_tuple_str_int)
out_typed[0]
out_typed[1]
for j in typed_variadic_tuple_generic_passthrough(untyped_tuple_str_int):
    j
for k in typed_variadic_tuple_generic_passthrough(typed_tuple_str_int):
    k
for l in typed_variadic_tuple_generic_passthrough(variadic_tuple_str):
    l
for m in typed_variadic_tuple_generic_passthrough(variadic_tuple_str_int):
    m
typed_fully_generic_passthrough(float)
for n in typed_fully_generic_passthrough(untyped_list_str):
    n
for o in typed_fully_generic_passthrough(typed_list_str):
    o
for p in typed_bound_generic_passthrough(untyped_list_str):
    p
for q in typed_bound_generic_passthrough(typed_list_str):
    q
for r in typed_quoted_return_generic_passthrough('something'):
    r
for s in typed_quoted_return_generic_passthrough(42):
    s
typed_quoted_input_generic_passthrough(('something',))
typed_quoted_input_generic_passthrough((42,))

class CustomList(List):

    def get_first(self):
        if False:
            return 10
        return self[0]
CustomList[str]()[0]
CustomList[str]().get_first()
typed_fully_generic_passthrough(CustomList[str]())[0]
typed_list_generic_passthrough(CustomList[str])[0]

def typed_bound_type_implicit_any_generic_passthrough(x: TType) -> TType:
    if False:
        while True:
            i = 10
    x
    return x

def typed_bound_type_any_generic_passthrough(x: TTypeAny) -> TTypeAny:
    if False:
        while True:
            i = 10
    x
    return x

class MyClass:
    pass

def my_func(a: str, b: int) -> float:
    if False:
        print('Hello World!')
    pass
typed_fully_generic_passthrough(MyClass)
typed_fully_generic_passthrough(MyClass())
typed_fully_generic_passthrough(my_func)
typed_bound_generic_passthrough(CustomList[str]())
typed_bound_generic_passthrough(42)
typed_bound_type_implicit_any_generic_passthrough(MyClass)
typed_bound_type_any_generic_passthrough(MyClass)
typed_bound_type_implicit_any_generic_passthrough(42)
typed_bound_type_any_generic_passthrough(42)

def decorator(fn: TCallable) -> TCallable:
    if False:
        print('Hello World!')
    pass

def will_be_decorated(the_param: complex) -> float:
    if False:
        return 10
    pass
is_decorated = decorator(will_be_decorated)
is_decorated
is_decorated(the_para)

class class_decorator_factory_plain:

    def __call__(self, func: T) -> T:
        if False:
            print('Hello World!')
        ...
class_decorator_factory_plain()
class_decorator_factory_plain()()
is_decorated_by_class_decorator_factory = class_decorator_factory_plain()(will_be_decorated)
is_decorated_by_class_decorator_factory
is_decorated_by_class_decorator_factory(the_par)

def decorator_factory_plain() -> Callable[[T], T]:
    if False:
        return 10
    pass
decorator_factory_plain()
decorator_factory_plain()()
decorator_factory_plain()(42)
is_decorated_by_plain_factory = decorator_factory_plain()(will_be_decorated)
is_decorated_by_plain_factory
is_decorated_by_plain_factory(the_par)

class class_decorator_factory_bound_callable:

    def __call__(self, func: TCallable) -> TCallable:
        if False:
            while True:
                i = 10
        ...
class_decorator_factory_bound_callable()
class_decorator_factory_bound_callable()()
is_decorated_by_class_bound_factory = class_decorator_factory_bound_callable()(will_be_decorated)
is_decorated_by_class_bound_factory
is_decorated_by_class_bound_factory(the_par)

def decorator_factory_bound_callable() -> Callable[[TCallable], TCallable]:
    if False:
        print('Hello World!')
    pass
decorator_factory_bound_callable()
decorator_factory_bound_callable()()
is_decorated_by_bound_factory = decorator_factory_bound_callable()(will_be_decorated)
is_decorated_by_bound_factory
is_decorated_by_bound_factory(the_par)

class That(Generic[T]):

    def __init__(self, items: List[Tuple[str, T]]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def get(self) -> T:
        if False:
            for i in range(10):
                print('nop')
        pass
inst = That([('abc', 2)])
inst.get()
from typing import Callable, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union, Sequence
K = TypeVar('K')
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
V = TypeVar('V')
just_float: float = 42.0
optional_float: Optional[float] = 42.0
list_of_ints: List[int] = [42]
list_of_floats: List[float] = [42.0]
list_of_optional_floats: List[Optional[float]] = [x or None for x in list_of_floats]
list_of_ints_and_strs: List[Union[int, str]] = [42, 'abc']

def list_t_to_list_t(the_list: List[T]) -> List[T]:
    if False:
        print('Hello World!')
    return the_list
x0 = list_t_to_list_t(list_of_ints)[0]
x0
for a in list_t_to_list_t(list_of_ints):
    a
x2 = list_t_to_list_t(list_of_ints_and_strs)[0]
x2
for z in list_t_to_list_t(list_of_ints_and_strs):
    z
list_of_int_type: List[Type[int]] = [int]

def list_optional_t_to_list_t(the_list: List[Optional[T]]) -> List[T]:
    if False:
        for i in range(10):
            print('nop')
    return [x for x in the_list if x is not None]
for xa in list_optional_t_to_list_t(list_of_optional_floats):
    xa
for xa1 in list_optional_t_to_list_t(list_of_floats):
    xa1

def optional_t_to_list_t(x: Optional[T]) -> List[T]:
    if False:
        i = 10
        return i + 15
    return [x] if x is not None else []
for xb in optional_t_to_list_t(optional_float):
    xb
for xb2 in optional_t_to_list_t(just_float):
    xb2

def optional_list_t_to_list_t(x: Optional[List[T]]) -> List[T]:
    if False:
        return 10
    return x if x is not None else []
optional_list_float: Optional[List[float]] = None
for xc in optional_list_t_to_list_t(optional_list_float):
    xc
for xc2 in optional_list_t_to_list_t(list_of_floats):
    xc2

def list_type_t_to_list_t(the_list: List[Type[T]]) -> List[T]:
    if False:
        while True:
            i = 10
    return [x() for x in the_list]
x1 = list_type_t_to_list_t(list_of_int_type)[0]
x1
for b in list_type_t_to_list_t(list_of_int_type):
    b

def list_t_to_list_tuple_t(the_list: List[T]) -> List[Tuple[T]]:
    if False:
        while True:
            i = 10
    return [(x,) for x in the_list]
x1t = list_t_to_list_tuple_t(list_of_ints)[0][0]
x1t
for c1 in list_t_to_list_tuple_t(list_of_ints):
    c1[0]
for (c2,) in list_t_to_list_tuple_t(list_of_ints):
    c2

def list_tuple_t_to_tuple_list_t(the_list: List[Tuple[T]]) -> Tuple[List[T], ...]:
    if False:
        for i in range(10):
            print('nop')
    return tuple((list(x) for x in the_list))
list_of_int_tuples: List[Tuple[int]] = [(x,) for x in list_of_ints]
for b in list_tuple_t_to_tuple_list_t(list_of_int_tuples):
    b[0]

def list_tuple_t_elipsis_to_tuple_list_t(the_list: List[Tuple[T, ...]]) -> Tuple[List[T], ...]:
    if False:
        return 10
    return tuple((list(x) for x in the_list))
list_of_int_tuple_elipsis: List[Tuple[int, ...]] = [tuple(list_of_ints)]
for b in list_tuple_t_elipsis_to_tuple_list_t(list_of_int_tuple_elipsis):
    b[0]

def foo(x: int) -> int:
    if False:
        print('Hello World!')
    return x
list_of_funcs: List[Callable[[int], int]] = [foo]

def list_func_t_to_list_func_type_t(the_list: List[Callable[[T], T]]) -> List[Callable[[Type[T]], T]]:
    if False:
        print('Hello World!')

    def adapt(func: Callable[[T], T]) -> Callable[[Type[T]], T]:
        if False:
            return 10

        def wrapper(typ: Type[T]) -> T:
            if False:
                while True:
                    i = 10
            return func(typ())
        return wrapper
    return [adapt(x) for x in the_list]
for b in list_func_t_to_list_func_type_t(list_of_funcs):
    b(int)

def bar(*a, **k) -> int:
    if False:
        while True:
            i = 10
    return len(a) + len(k)
list_of_funcs_2: List[Callable[..., int]] = [bar]

def list_func_t_passthrough(the_list: List[Callable[..., T]]) -> List[Callable[..., T]]:
    if False:
        while True:
            i = 10
    return the_list
for b in list_func_t_passthrough(list_of_funcs_2):
    b(None, x='x')
mapping_int_str: Dict[int, str] = {42: 'a'}

def invert_mapping(mapping: Mapping[K, V]) -> Mapping[V, K]:
    if False:
        return 10
    return {v: k for (k, v) in mapping.items()}
invert_mapping(mapping_int_str)['a']

def first(iterable: Iterable[T]) -> T:
    if False:
        return 10
    return next(iter(iterable))
first(mapping_int_str)
first('abc')
some_str: str = NotImplemented
first(some_str)
annotated: List[Callable[[Sequence[float]], int]] = [len]
first(annotated)()

def values(mapping: Mapping[int, T]) -> List[T]:
    if False:
        i = 10
        return i + 15
    return list(mapping.values())
values(mapping_int_str)[0]
x2 = values(mapping_int_str)[0]
x2
for b in values(mapping_int_str):
    b
list_ints: List[int] = [42]

class CustomGeneric(Generic[T_co]):

    def __init__(self, val: T_co) -> None:
        if False:
            while True:
                i = 10
        self.val = val

def custom(x: CustomGeneric[T]) -> T:
    if False:
        return 10
    return x.val
custom_instance: CustomGeneric[int] = CustomGeneric(42)
custom(custom_instance)
x3 = custom(custom_instance)
x3

def wrap_custom(iterable: Iterable[T]) -> List[CustomGeneric[T]]:
    if False:
        return 10
    return [CustomGeneric(x) for x in iterable]
wrap_custom(list_ints)[0].val
x4 = wrap_custom(list_ints)[0]
x4.val
for x5 in wrap_custom(list_ints):
    x5.val
list_custom_instances: List[CustomGeneric[int]] = [CustomGeneric(42)]

def unwrap_custom(iterable: Iterable[CustomGeneric[T]]) -> List[T]:
    if False:
        print('Hello World!')
    return [x.val for x in iterable]
unwrap_custom(list_custom_instances)[0]
x6 = unwrap_custom(list_custom_instances)[0]
x6
for x7 in unwrap_custom(list_custom_instances):
    x7
for xc in unwrap_custom([CustomGeneric(s) for s in 'abc']):
    xc
for xg in unwrap_custom((CustomGeneric(s) for s in 'abc')):
    xg
custom_instance_list_int: CustomGeneric[List[int]] = CustomGeneric([42])

def unwrap_custom2(instance: CustomGeneric[Iterable[T]]) -> List[T]:
    if False:
        for i in range(10):
            print('nop')
    return list(instance.val)
unwrap_custom2(custom_instance_list_int)[0]
x8 = unwrap_custom2(custom_instance_list_int)[0]
x8
for x9 in unwrap_custom2(custom_instance_list_int):
    x9

class Specialised(Mapping[int, str]):
    pass
specialised_instance: Specialised = NotImplemented
first(specialised_instance)
values(specialised_instance)[0]

class ChildOfSpecialised(Specialised):
    pass
child_of_specialised_instance: ChildOfSpecialised = NotImplemented
first(child_of_specialised_instance)
values(child_of_specialised_instance)[0]

class CustomPartialGeneric1(Mapping[str, T]):
    pass
custom_partial1_instance: CustomPartialGeneric1[int] = NotImplemented
first(custom_partial1_instance)
custom_partial1_unbound_instance: CustomPartialGeneric1 = NotImplemented
first(custom_partial1_unbound_instance)

class CustomPartialGeneric2(Mapping[T, str]):
    pass
custom_partial2_instance: CustomPartialGeneric2[int] = NotImplemented
first(custom_partial2_instance)
values(custom_partial2_instance)[0]
custom_partial2_unbound_instance: CustomPartialGeneric2 = NotImplemented
first(custom_partial2_unbound_instance)
values(custom_partial2_unbound_instance)[0]
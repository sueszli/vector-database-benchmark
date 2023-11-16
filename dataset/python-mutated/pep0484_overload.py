from typing import List, Dict, overload, Tuple, TypeVar
lst: list
list_alias: List
list_str: List[str]
list_int: List[int]

@overload
def overload_f2(value: List) -> str:
    if False:
        return 10
    ...

@overload
def overload_f2(value: Dict) -> int:
    if False:
        return 10
    ...
overload_f2([''])
overload_f2({1.0: 1.0})
overload_f2(lst)
overload_f2(list_alias)
overload_f2(list_str)

@overload
def overload_f3(value: list) -> str:
    if False:
        i = 10
        return i + 15
    ...

@overload
def overload_f3(value: dict) -> float:
    if False:
        for i in range(10):
            print('nop')
    ...
overload_f3([''])
overload_f3({1.0: 1.0})
overload_f3(lst)
overload_f3(list_alias)
overload_f3(list_str)

@overload
def overload_f1(value: List[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def overload_f1(value: Dict[str, str]) -> Dict[str, str]:
    if False:
        return 10
    ...

def overload_f1():
    if False:
        while True:
            i = 10
    pass
overload_f1([''])
overload_f1(1)
overload_f1({'': ''})
overload_f1(lst)
overload_f1(list_alias)
overload_f1(list_str)
overload_f1(list_int)
T = TypeVar('T')

@overload
def broken_f1(value: 1) -> str:
    if False:
        i = 10
        return i + 15
    ...

@overload
def broken_f1(value: Tuple[T]) -> Tuple[T]:
    if False:
        i = 10
        return i + 15
    ...
tup: Tuple[float]
broken_f1(broken_f1(tup))[0]
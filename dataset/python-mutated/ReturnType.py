from typing import List, Union

def A_no_return():
    if False:
        i = 10
        return i + 15
    pass

def B_none_return() -> None:
    if False:
        print('Hello World!')
    pass

def C_simple_return() -> int:
    if False:
        while True:
            i = 10
    return 42

def D_parameterized_return() -> List[int]:
    if False:
        while True:
            i = 10
    return []

def E_union_return() -> Union[int, float]:
    if False:
        return 10
    return 42

def F_stringified_return() -> 'int | float':
    if False:
        for i in range(10):
            print('nop')
    return 42

class Unknown:
    pass

def G_unknown_return() -> Unknown:
    if False:
        print('Hello World!')
    return Unknown()

def H_invalid_return() -> 'list[int':
    if False:
        i = 10
        return i + 15
    pass
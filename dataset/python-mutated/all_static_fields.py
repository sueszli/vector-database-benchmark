from dataclasses import dataclass
from typing import Optional, Union, List, Type
'\nThe .pysa file adds `TaintSink[Test, ParameterPath[_.all_static_fields()]]` on\nall parameters of functions. See `.models` file for the results.\n'

class RegularClass:

    def __init__(self, a: int, b: str) -> None:
        if False:
            print('Hello World!')
        self.a = a
        self.b = b

def parameter_sink_regular_class(parameter: RegularClass) -> None:
    if False:
        return 10
    pass

def return_source_regular_class() -> RegularClass:
    if False:
        return 10
    return RegularClass(a=0, b='')

def parameter_sink_optional_regular_class(parameter: Optional[RegularClass]) -> None:
    if False:
        return 10
    pass

def return_source_optional_regular_class() -> Optional[RegularClass]:
    if False:
        i = 10
        return i + 15
    return None

@dataclass
class Dataclass:
    c: int
    d: str

def parameter_sink_dataclass(parameter: Dataclass) -> None:
    if False:
        return 10
    pass

def return_source_dataclass() -> Dataclass:
    if False:
        for i in range(10):
            print('nop')
    return Dataclass(c=0, d='')

def parameter_sink_optional_dataclass(parameter: Optional[Dataclass]) -> None:
    if False:
        while True:
            i = 10
    pass

def return_source_optional_dataclass() -> Optional[Dataclass]:
    if False:
        i = 10
        return i + 15
    return None

def parameter_sink_union_dataclass_regular(parameter: Union[Dataclass, RegularClass]) -> None:
    if False:
        print('Hello World!')
    pass

def return_source_union_dataclass_regular() -> Union[Dataclass, RegularClass]:
    if False:
        while True:
            i = 10
    return RegularClass(a=0, b='')

def parameter_sink_builtin_parameters(x: int, y: str, z: List[int], t: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def return_source_builtin_int() -> int:
    if False:
        i = 10
        return i + 15
    return 0

def return_source_builtin_list() -> List[int]:
    if False:
        while True:
            i = 10
    return []

def parameter_sink_unnannotated(x) -> None:
    if False:
        print('Hello World!')
    pass

def return_source_unnannotated():
    if False:
        for i in range(10):
            print('nop')
    return 0

class A:
    """Test doc string"""

    def __init__(self, a: int) -> None:
        if False:
            i = 10
            return i + 15
        self.a = a

class B(A):
    pass

class C(B):

    def __init__(self, a: int, c: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__(a)
        self.c = c

class D(A):
    d: int

    def __init__(self, a: int, d: int) -> None:
        if False:
            return 10
        super().__init__(a)
        self.d = d

def parameter_sink_b(parameter: B) -> None:
    if False:
        while True:
            i = 10
    pass

def return_source_b() -> B:
    if False:
        while True:
            i = 10
    return C(a=0, c=0)

def parameter_sink_c(parameter: C) -> None:
    if False:
        return 10
    pass

def return_source_c() -> C:
    if False:
        i = 10
        return i + 15
    return C(a=0, c=0)

def parameter_sink_d(parameter: D) -> None:
    if False:
        while True:
            i = 10
    pass

def return_source_d() -> D:
    if False:
        while True:
            i = 10
    return D(a=0, d=0)

def parameter_sink_union_c_d(parameter: Union[C, D]) -> None:
    if False:
        while True:
            i = 10
    pass

def return_source_union_c_d() -> Union[C, D]:
    if False:
        for i in range(10):
            print('nop')
    return D(a=0, d=0)

def parameter_sink_type(parameter: Type[A]) -> None:
    if False:
        print('Hello World!')
    pass

def return_source_type() -> Type[A]:
    if False:
        i = 10
        return i + 15
    return A

class Empty:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

def parameter_sink_empty(parameter: Empty) -> None:
    if False:
        print('Hello World!')
    pass

def return_source_empty() -> Empty:
    if False:
        while True:
            i = 10
    return Empty()

async def return_source_async() -> bool:
    return True
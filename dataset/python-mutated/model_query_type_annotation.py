from typing import Annotated, Dict, List

def test1_f1(taint_1: int, taint_2: str, taint_3: float):
    if False:
        print('Hello World!')
    pass

def test1_f2(no_taint_1: List[str], no_taint_2: int):
    if False:
        for i in range(10):
            print('nop')
    pass

class Test2_T1:
    ...

class Test2_T2:
    ...

class Test2_Foo:
    ...

class Test2_C:

    def test2_f1(self, taint_1: Test2_T1, taint_2: Test2_T2, no_taint_1: Test2_Foo):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test2_f2(self, taint_1: Dict[int, Test2_T1]):
        if False:
            print('Hello World!')
        pass

    def test2_f3(self, no_taint_1: int, no_taint_2: str):
        if False:
            for i in range(10):
                print('nop')
        pass

def test3_f1(taint_1: Annotated[str, 'foo'], no_taint_1: str, taint_2: Annotated[int, 'bar']):
    if False:
        for i in range(10):
            print('nop')
    pass

def test3_f2(no_taint_1: List[Annotated[str, 'foo']], no_taint_2: int):
    if False:
        for i in range(10):
            print('nop')
    pass

def test4_taint_1(x) -> str:
    if False:
        return 10
    pass

def test4_no_taint_1(x) -> int:
    if False:
        for i in range(10):
            print('nop')
    pass

class Test5_T1:
    ...

class Test5_T2:
    ...

class Test5_Foo:
    ...

def test5_taint_1(x) -> Test5_T1:
    if False:
        return 10
    pass

def test5_taint_2(x) -> Test5_T2:
    if False:
        i = 10
        return i + 15
    pass

def test5_no_taint_1(x) -> Test5_Foo:
    if False:
        return 10
    pass

class Test6_C:

    def test6_taint_1(self, x) -> Annotated[str, 'foo']:
        if False:
            return 10
        pass

    def test6_taint_2(self, x) -> Annotated[List[str], 'bar']:
        if False:
            return 10
        pass

    def test6_no_taint_1(self, x) -> str:
        if False:
            while True:
                i = 10
        pass

class Test7_C:
    taint_1: int = 0
    taint_2: int = 0
    no_taint_1: List[int] = []
    no_taint_2: str = ''

class Test8_C:
    taint_1: str = ''
    taint_2: List[str] = []
    no_taint_1: List[int] = []
    no_taint_2: int = 0

class Test9_C:
    taint_1: Annotated[str, 'foo'] = ''
    taint_2: Annotated[int, 'bar'] = 0
    no_taint_1: List[int] = []
    no_taint_2: int = 0
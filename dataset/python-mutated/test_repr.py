from enum import Enum
import strawberry

def test_repr_type():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class MyType:
        s: str
        i: int
        b: bool
        f: float
        id: strawberry.ID
    assert repr(MyType(s='a', i=1, b=True, f=3.2, id='123')) == "test_repr_type.<locals>.MyType(s='a', i=1, b=True, f=3.2, id='123')"

def test_repr_enum():
    if False:
        return 10

    @strawberry.enum()
    class Test(Enum):
        A = 1
        B = 2
        C = 3
    assert repr(Test(1)) == '<Test.A: 1>'
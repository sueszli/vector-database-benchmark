from enum import Enum
import strawberry
from strawberry.annotation import StrawberryAnnotation

def test_basic():
    if False:
        print('Hello World!')

    @strawberry.enum
    class NumaNuma(Enum):
        MA = 'ma'
        I = 'i'
        A = 'a'
        HI = 'hi'
    annotation = StrawberryAnnotation(NumaNuma)
    resolved = annotation.resolve()
    assert resolved is NumaNuma._enum_definition
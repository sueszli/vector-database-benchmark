from dataclasses import dataclass
from dataclasses import InitVar

@dataclass
class Foo:
    init_only: InitVar[int]
    real_attr: int

def test_demonstrate():
    if False:
        print('Hello World!')
    assert Foo(1, 2) == Foo(1, 3)
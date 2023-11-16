from dataclasses import dataclass
from dataclasses import field

def test_dataclasses() -> None:
    if False:
        while True:
            i = 10

    @dataclass
    class SimpleDataObject:
        field_a: int = field()
        field_b: str = field()
    left = SimpleDataObject(1, 'b')
    right = SimpleDataObject(1, 'c')
    assert left == right
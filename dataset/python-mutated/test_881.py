import pickle
from ...types.enum import Enum

class PickleEnum(Enum):
    A = 'a'
    B = 1

def test_enums_pickling():
    if False:
        i = 10
        return i + 15
    a = PickleEnum.A
    pickled = pickle.dumps(a)
    restored = pickle.loads(pickled)
    assert type(a) is type(restored)
    assert a == restored
    assert a.value == restored.value
    assert a.name == restored.name
    b = PickleEnum.B
    pickled = pickle.dumps(b)
    restored = pickle.loads(pickled)
    assert type(a) is type(restored)
    assert b == restored
    assert b.value == restored.value
    assert b.name == restored.name
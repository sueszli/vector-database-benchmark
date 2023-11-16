import pytest
from libqtile.ipc import _IPC

def test_ipc_json_encoder_supports_sets():
    if False:
        print('Hello World!')
    serialized = _IPC.pack({'foo': set()}, is_json=True)
    assert serialized == b'{"foo": []}'

def test_ipc_json_throws_error_on_unsupported_field():
    if False:
        while True:
            i = 10

    class NonSerializableType:
        ...
    with pytest.raises(ValueError, match="Tried to JSON serialize unsupported type <class 'test.test_ipc.test_ipc_json_throws_error_on_unsupported_field.<locals>.NonSerializableType'>.*"):
        _IPC.pack({'foo': NonSerializableType()}, is_json=True)

def test_ipc_marshall_error_on_unsupported_field():
    if False:
        return 10

    class NonSerializableType:
        ...
    with pytest.raises(ValueError, match='unmarshallable object'):
        _IPC.pack({'foo': NonSerializableType()})
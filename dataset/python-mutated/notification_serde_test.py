import pytest
from pytest import FixtureRequest
import syft as sy

@pytest.mark.parametrize('obj_name', ['mock_notification', 'mock_create_notification'])
def test_notification_serde(obj_name: str, request: FixtureRequest) -> None:
    if False:
        return 10
    obj = request.getfixturevalue(obj_name)
    ser_data = sy.serialize(obj, to_bytes=True)
    assert isinstance(ser_data, bytes)
    deser_data = sy.deserialize(ser_data, from_bytes=True)
    assert isinstance(deser_data, type(obj))
    assert deser_data == obj
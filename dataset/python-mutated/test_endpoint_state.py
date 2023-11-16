import sys
from typing import Any, Tuple
from unittest.mock import Mock, patch
import pytest
from ray.serve._private.endpoint_state import EndpointState

class MockKVStore:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.store = dict()

    def put(self, key: str, val: Any) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        self.store[key] = val
        return True

    def get(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        return self.store.get(key, None)

    def delete(self, key: str) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        if key in self.store:
            del self.store[key]
            return True
        return False

@pytest.fixture
def mock_endpoint_state() -> Tuple[EndpointState, Mock]:
    if False:
        for i in range(10):
            print('nop')
    with patch('ray.serve._private.long_poll.LongPollHost') as mock_long_poll:
        endpoint_state = EndpointState(kv_store=MockKVStore(), long_poll_host=mock_long_poll)
        yield endpoint_state

def test_is_ready_for_shutdown(mock_endpoint_state):
    if False:
        for i in range(10):
            print('nop')
    'Test `is_ready_for_shutdown()` returns the correct state.\n\n    Before shutting down endpoint `is_ready_for_shutdown()` should return False.\n    After shutting down endpoint `is_ready_for_shutdown()` should return True.\n    '
    endpoint_state = mock_endpoint_state
    endpoint_state._checkpoint()
    assert not endpoint_state.is_ready_for_shutdown()
    endpoint_state.shutdown()
    assert endpoint_state.is_ready_for_shutdown()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))
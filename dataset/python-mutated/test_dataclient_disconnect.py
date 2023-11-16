import os
import time
from unittest.mock import Mock, patch
import pytest
from ray.util.client.ray_client_helpers import ray_start_client_server

def test_dataclient_disconnect_on_request():
    if False:
        for i in range(10):
            print('nop')
    with patch.dict(os.environ, {'RAY_CLIENT_RECONNECT_GRACE_PERIOD': '5'}), ray_start_client_server() as ray:
        assert ray.is_connected()

        @ray.remote
        def f():
            if False:
                return 10
            return 42
        assert ray.get(f.remote()) == 42
        with pytest.raises(ConnectionError):
            ray.worker.data_client._blocking_send(Mock())
        assert not ray.is_connected()
        time.sleep(5)
        connection_data = ray.connect('localhost:50051')
        assert connection_data['num_clients'] == 1
        assert ray.get(f.remote()) == 42

def test_dataclient_disconnect_before_request():
    if False:
        for i in range(10):
            print('nop')
    with patch.dict(os.environ, {'RAY_CLIENT_RECONNECT_GRACE_PERIOD': '5'}), ray_start_client_server() as ray:
        assert ray.is_connected()

        @ray.remote
        def f():
            if False:
                print('Hello World!')
            return 42
        assert ray.get(f.remote()) == 42
        ray.worker.data_client.request_queue.put(Mock())
        with pytest.raises(ConnectionError):
            ray.get(f.remote())
        with pytest.raises(ConnectionError, match='Ray client has already been disconnected'):
            ray.get(f.remote())
        assert not ray.is_connected()
        time.sleep(5)
        connection_data = ray.connect('localhost:50051')
        assert connection_data['num_clients'] == 1
        assert ray.get(f.remote()) == 42
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))
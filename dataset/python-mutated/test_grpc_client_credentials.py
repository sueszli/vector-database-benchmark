import pytest
import grpc
from ray.util.client.worker import Worker

class Credentials(grpc.ChannelCredentials):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

def test_grpc_client_credentials_are_passed_to_channel(monkeypatch):
    if False:
        print('Hello World!')

    class Stop(Exception):

        def __init__(self, credentials):
            if False:
                for i in range(10):
                    print('nop')
            self.credentials = credentials

    class MockChannel:

        def __init__(self, conn_str, credentials, options, compression):
            if False:
                print('Hello World!')
            self.credentials = credentials

        def subscribe(self, f):
            if False:
                print('Hello World!')
            raise Stop(self.credentials)

    def mock_secure_channel(conn_str, credentials, options=None, compression=None):
        if False:
            print('Hello World!')
        return MockChannel(conn_str, credentials, options, compression)
    monkeypatch.setattr(grpc, 'secure_channel', mock_secure_channel)
    with pytest.raises(Stop) as stop:
        Worker(secure=False, _credentials=Credentials('test'))
    assert stop.value.credentials.name == 'test'
    with pytest.raises(Stop) as stop:
        Worker(secure=True, _credentials=Credentials('test'))
    assert stop.value.credentials.name == 'test'

def test_grpc_client_credentials_are_generated(monkeypatch):
    if False:
        i = 10
        return i + 15

    class Stop(Exception):

        def __init__(self, result):
            if False:
                return 10
            self.result = result

    def mock_gen_credentials():
        if False:
            while True:
                i = 10
        raise Stop('ssl_channel_credentials called')
    monkeypatch.setattr(grpc, 'ssl_channel_credentials', mock_gen_credentials)
    with pytest.raises(Stop) as stop:
        Worker(secure=True)
    assert stop.value.result == 'ssl_channel_credentials called'
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))
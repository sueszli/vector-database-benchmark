from unittest import mock
import pytest
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import HTTPError, ReadTimeoutError
from sentry.utils.snuba import RetrySkipTimeout

class FakeConnectionPool(HTTPConnectionPool):

    def __init__(self, connection, **kwargs):
        if False:
            i = 10
            return i + 15
        self.connection = connection
        super().__init__(**kwargs)

    def _new_conn(self):
        if False:
            print('Hello World!')
        return self.connection

def test_retries():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that, even if I set up 5 retries, there is only one request\n    made since it times out.\n    '
    connection_mock = mock.Mock()
    snuba_pool = FakeConnectionPool(connection=connection_mock, host='www.test.com', port=80, retries=RetrySkipTimeout(total=5, allowed_methods={'GET', 'POST'}), timeout=30, maxsize=10)
    connection_mock.request.side_effect = ReadTimeoutError(snuba_pool, 'test.com', 'Timeout')
    with pytest.raises(HTTPError):
        snuba_pool.urlopen('POST', '/query', body='{}')
    assert connection_mock.request.call_count == 1
import pytest
from mock import Mock, patch
from nameko.utils.retry import retry

@pytest.fixture
def tracker():
    if False:
        while True:
            i = 10
    return Mock()

@pytest.fixture(autouse=True)
def mock_sleep():
    if False:
        print('Hello World!')
    with patch('nameko.utils.retry.sleep') as patched:

        def total():
            if False:
                print('Hello World!')
            return sum((delay for ((delay,), _) in patched.call_args_list))
        patched.total = total
        yield patched

def test_without_arguments(tracker, mock_sleep):
    if False:
        while True:
            i = 10

    @retry
    def fn():
        if False:
            i = 10
            return i + 15
        tracker()
        raise ValueError()
    with pytest.raises(ValueError):
        fn()
    assert tracker.call_count > 1

@pytest.mark.parametrize('exceptions', [ValueError, (ValueError,)])
def test_retry_for_exceptions(exceptions, tracker):
    if False:
        while True:
            i = 10
    threshold = 10

    @retry(for_exceptions=exceptions, max_attempts=float('inf'))
    def fn():
        if False:
            print('Hello World!')
        tracker()
        if tracker.call_count < threshold:
            raise ValueError()
        else:
            raise KeyError()
    with pytest.raises(KeyError):
        fn()
    assert tracker.call_count == threshold

class TestRetries(object):

    def test_retry_limit(self, tracker):
        if False:
            i = 10
            return i + 15
        max_attempts = 5

        @retry(max_attempts=max_attempts)
        def fn():
            if False:
                print('Hello World!')
            tracker()
            raise ValueError()
        with pytest.raises(ValueError):
            fn()
        assert tracker.call_count == 1 + max_attempts

    def test_retry_forever(self, tracker):
        if False:
            return 10
        threshold = 10

        @retry(max_attempts=None)
        def fn():
            if False:
                while True:
                    i = 10
            tracker()
            if tracker.call_count == threshold:
                return threshold
            else:
                raise ValueError()
        assert fn() == threshold
        assert tracker.call_count == threshold

class TestDelay(object):

    def test_fixed_delay(self, tracker, mock_sleep):
        if False:
            for i in range(10):
                print('nop')
        max_attempts = 5
        delay = 1

        @retry(max_attempts=max_attempts, delay=delay)
        def fn():
            if False:
                i = 10
                return i + 15
            tracker()
            raise ValueError()
        with pytest.raises(ValueError):
            fn()
        total_delay = mock_sleep.total()
        assert total_delay == delay * max_attempts

    def test_backoff(self, tracker, mock_sleep):
        if False:
            i = 10
            return i + 15
        max_attempts = 5
        delay = 2
        backoff = 3

        @retry(max_attempts=max_attempts, delay=delay, backoff=backoff)
        def fn():
            if False:
                return 10
            tracker()
            raise ValueError()
        with pytest.raises(ValueError):
            fn()
        total_delay = mock_sleep.total()
        assert total_delay == sum((delay * backoff ** attempt for attempt in range(1, max_attempts + 1)))

    def test_max_delay(self, tracker, mock_sleep):
        if False:
            return 10
        max_attempts = 5
        delay = 1
        backoff = 2
        max_delay = delay

        @retry(max_attempts=max_attempts, delay=delay, backoff=backoff, max_delay=max_delay)
        def fn():
            if False:
                print('Hello World!')
            tracker()
            raise ValueError()
        with pytest.raises(ValueError):
            fn()
        total_delay = mock_sleep.total()
        assert total_delay == delay * max_attempts
import logging
import pytest
from django.utils.timezone import now, timedelta
from awx.main.analytics.core import calculate_collection_interval
logger = logging.getLogger('awx.main.analytics')
logger.propagate = True
epsilon = timedelta(minutes=1)

class TestIntervalWithSinceAndUntil:

    @pytest.mark.parametrize('gather', [None, 2, 6])
    def test_ok(self, caplog, settings, gather):
        if False:
            while True:
                i = 10
        _now = now()
        _prior = _now - timedelta(weeks=gather) if gather is not None else None
        until = _now
        since = until - timedelta(weeks=3)
        settings.AUTOMATION_ANALYTICS_LAST_GATHER = _prior
        (new_since, new_until, last_gather) = calculate_collection_interval(since, until)
        assert new_since == since
        assert new_until == until
        expected = 1 if gather and gather > 4 else 0
        assert len(caplog.records) == expected
        assert sum((1 for msg in caplog.messages if 'more than 4 weeks prior' in msg)) == expected
        assert last_gather is not None
        assert abs(until - last_gather - timedelta(weeks=gather if gather and gather <= 4 else 4)) < epsilon

    def test_both_in_future(self, caplog):
        if False:
            while True:
                i = 10
        since = now() + timedelta(weeks=1)
        until = since + timedelta(weeks=1)
        with pytest.raises(ValueError):
            calculate_collection_interval(since, until)
        assert len(caplog.records) == 3
        assert sum((1 for msg in caplog.messages if 'is in the future' in msg)) == 2
        assert sum((1 for msg in caplog.messages if 'later than the end' in msg)) == 1

    def test_until_in_future(self, caplog):
        if False:
            return 10
        _now = now()
        since = _now - timedelta(weeks=1)
        until = _now + timedelta(weeks=1)
        (new_since, new_until, _) = calculate_collection_interval(since, until)
        assert new_since == since
        assert abs(_now - new_until) < epsilon
        assert len(caplog.records) == 1
        assert sum((1 for msg in caplog.messages if 'is in the future' in msg)) == 1

    def test_interval_too_large(self, caplog):
        if False:
            i = 10
            return i + 15
        until = now()
        since = until - timedelta(weeks=5)
        (new_since, new_until, _) = calculate_collection_interval(since, until)
        assert new_since == since
        assert new_until == since + timedelta(weeks=4)
        assert len(caplog.records) == 1
        assert sum((1 for msg in caplog.messages if 'greater than 4 weeks from start' in msg)) == 1

    def test_reversed(self, caplog):
        if False:
            print('Hello World!')
        since = now()
        until = since - timedelta(weeks=3)
        with pytest.raises(ValueError):
            calculate_collection_interval(since, until)
        assert len(caplog.records) == 1
        assert sum((1 for msg in caplog.messages if 'later than the end' in msg)) == 1

class TestIntervalWithSinceOnly:

    @pytest.mark.parametrize('gather', [None, 2, 6])
    def test_ok(self, caplog, settings, gather):
        if False:
            return 10
        _now = now()
        _prior = _now - timedelta(weeks=gather) if gather is not None else None
        since = _now - timedelta(weeks=2)
        until = None
        settings.AUTOMATION_ANALYTICS_LAST_GATHER = _prior
        (new_since, new_until, last_gather) = calculate_collection_interval(since, until)
        assert new_since == since
        assert abs(new_until - _now) < epsilon
        expected = 1 if gather and gather > 4 else 0
        assert len(caplog.records) == expected
        assert sum((1 for msg in caplog.messages if 'more than 4 weeks prior' in msg)) == expected
        assert last_gather is not None
        assert abs(_now - last_gather - timedelta(weeks=gather if gather and gather <= 4 else 4)) < epsilon

    def test_since_more_than_4_weeks_before_now(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        since = now() - timedelta(weeks=5)
        until = None
        (new_since, new_until, last_gather) = calculate_collection_interval(since, until)
        assert new_since == since
        assert new_until == since + timedelta(weeks=4)
        assert len(caplog.records) == 0

    def test_since_in_future(self, caplog):
        if False:
            return 10
        since = now() + timedelta(weeks=1)
        until = None
        with pytest.raises(ValueError):
            calculate_collection_interval(since, until)
        assert len(caplog.records) == 2
        assert sum((1 for msg in caplog.messages if 'is in the future' in msg)) == 1
        assert sum((1 for msg in caplog.messages if 'later than the end' in msg)) == 1

class TestIntervalWithUntilOnly:

    @pytest.mark.parametrize('gather', [None, 2, 6])
    def test_ok(self, caplog, settings, gather):
        if False:
            print('Hello World!')
        _now = now()
        _prior = _now - timedelta(weeks=gather) if gather is not None else None
        since = None
        until = _now - timedelta(weeks=1)
        settings.AUTOMATION_ANALYTICS_LAST_GATHER = _prior
        (new_since, new_until, last_gather) = calculate_collection_interval(since, until)
        assert new_since is None
        assert new_until == until
        assert last_gather is not None
        assert abs(_now - last_gather - timedelta(weeks=gather if gather and gather <= 5 else 5)) < epsilon
        expected = 1 if gather and gather > 5 else 0
        assert len(caplog.records) == expected
        assert sum((1 for msg in caplog.messages if 'more than 4 weeks prior' in msg)) == expected

    def test_until_in_future(self, caplog):
        if False:
            return 10
        _now = now()
        since = None
        until = _now + timedelta(weeks=1)
        (new_since, new_until, _) = calculate_collection_interval(since, until)
        assert new_since is None
        assert abs(new_until - _now) < epsilon
        assert len(caplog.records) == 1
        assert sum((1 for msg in caplog.messages if 'is in the future' in msg)) == 1

class TestIntervalWithNoParams:

    @pytest.mark.parametrize('gather', [None, 2, 6])
    def test_ok(self, caplog, settings, gather):
        if False:
            for i in range(10):
                print('nop')
        _now = now()
        _prior = _now - timedelta(weeks=gather) if gather is not None else None
        (since, until) = (None, None)
        settings.AUTOMATION_ANALYTICS_LAST_GATHER = _prior
        (new_since, new_until, last_gather) = calculate_collection_interval(since, until)
        assert new_since is None
        assert abs(new_until - _now) < epsilon
        assert last_gather is not None
        assert abs(_now - last_gather - timedelta(weeks=gather if gather and gather <= 4 else 4)) < epsilon
        expected = 1 if gather and gather > 4 else 0
        assert len(caplog.records) == expected
        assert sum((1 for msg in caplog.messages if 'more than 4 weeks prior' in msg)) == expected
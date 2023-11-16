import gevent
import pytest

@pytest.fixture(autouse=True)
def fast_wait(monkeypatch):
    if False:
        return 10
    'Stub out gevent calls that take timeouts to wait briefly.\n\n    In production one may want to wait a bit having no work to do to\n    avoid spinning, but during testing this adds quite a bit of time.\n\n    '
    old_sleep = gevent.sleep
    old_joinall = gevent.joinall
    old_killall = gevent.killall

    def fast_wait(tm):
        if False:
            print('Hello World!')
        return old_sleep(0.1)

    def fast_joinall(*args, **kwargs):
        if False:
            while True:
                i = 10
        if 'timeout' in kwargs:
            kwargs['timeout'] = 0.1
        return old_joinall(*args, **kwargs)

    def fast_killall(*args, **kwargs):
        if False:
            return 10
        if 'timeout' in kwargs:
            kwargs['timeout'] = 0.1
        return old_killall(*args, **kwargs)
    monkeypatch.setattr(gevent, 'sleep', fast_wait)
    monkeypatch.setattr(gevent, 'joinall', fast_joinall)
    monkeypatch.setattr(gevent, 'killall', fast_killall)
from functools import wraps
from threading import Event
from time import sleep, time
from tqdm import TMonitor, tqdm, trange
from .tests_tqdm import StringIO, closing, importorskip, patch_lock, skip

class Time(object):
    """Fake time class class providing an offset"""
    offset = 0

    @classmethod
    def reset(cls):
        if False:
            return 10
        'zeroes internal offset'
        cls.offset = 0

    @classmethod
    def time(cls):
        if False:
            return 10
        'time.time() + offset'
        return time() + cls.offset

    @staticmethod
    def sleep(dur):
        if False:
            print('Hello World!')
        'identical to time.sleep()'
        sleep(dur)

    @classmethod
    def fake_sleep(cls, dur):
        if False:
            i = 10
            return i + 15
        'adds `dur` to internal offset'
        cls.offset += dur
        sleep(1e-06)

class FakeEvent(Event):
    """patched `threading.Event` where `wait()` uses `Time.fake_sleep()`"""

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        'uses Time.fake_sleep'
        if timeout is not None:
            Time.fake_sleep(timeout)
        return self.is_set()

def patch_sleep(func):
    if False:
        return 10
    'Temporarily makes TMonitor use Time.fake_sleep'

    @wraps(func)
    def inner(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'restores TMonitor on completion regardless of Exceptions'
        TMonitor._test['time'] = Time.time
        TMonitor._test['Event'] = FakeEvent
        if tqdm.monitor:
            assert not tqdm.monitor.get_instances()
            tqdm.monitor.exit()
            del tqdm.monitor
            tqdm.monitor = None
        try:
            return func(*args, **kwargs)
        finally:
            tqdm.monitor_interval = 10
            if tqdm.monitor:
                assert not tqdm.monitor.get_instances()
                tqdm.monitor.exit()
                del tqdm.monitor
                tqdm.monitor = None
            TMonitor._test.pop('Event')
            TMonitor._test.pop('time')
    return inner

def cpu_timify(t, timer=Time):
    if False:
        for i in range(10):
            print('nop')
    'Force tqdm to use the specified timer instead of system-wide time'
    t._time = timer.time
    t._sleep = timer.fake_sleep
    t.start_t = t.last_print_t = t._time()
    return timer

class FakeTqdm(object):
    _instances = set()
    get_lock = tqdm.get_lock

def incr(x):
    if False:
        print('Hello World!')
    return x + 1

def incr_bar(x):
    if False:
        i = 10
        return i + 15
    with closing(StringIO()) as our_file:
        for _ in trange(x, lock_args=(False,), file=our_file):
            pass
    return incr(x)

@patch_sleep
def test_monitor_thread():
    if False:
        return 10
    'Test dummy monitoring thread'
    monitor = TMonitor(FakeTqdm, 10)
    assert monitor.report()
    monitor.exit()
    assert not monitor.report()
    assert not monitor.is_alive()
    del monitor

@patch_sleep
def test_monitoring_and_cleanup():
    if False:
        i = 10
        return i + 15
    'Test for stalled tqdm instance and monitor deletion'
    maxinterval = tqdm.monitor_interval
    assert maxinterval == 10
    total = 1000
    with closing(StringIO()) as our_file:
        with tqdm(total=total, file=our_file, miniters=500, mininterval=0.1, maxinterval=maxinterval) as t:
            cpu_timify(t, Time)
            Time.fake_sleep(maxinterval / 10)
            t.update(500)
            assert t.miniters <= 500
            Time.fake_sleep(maxinterval)
            t.update(1)
            timeend = Time.time()
            while not (t.monitor.woken >= timeend and t.miniters == 1):
                Time.fake_sleep(1)
            assert t.miniters == 1
            Time.fake_sleep(maxinterval)
            t.update(2)
            timeend = Time.time()
            while t.monitor.woken < timeend:
                Time.fake_sleep(1)
            assert t.miniters == 1

@patch_sleep
def test_monitoring_multi():
    if False:
        return 10
    'Test on multiple bars, one not needing miniters adjustment'
    maxinterval = tqdm.monitor_interval
    assert maxinterval == 10
    total = 1000
    with closing(StringIO()) as our_file:
        with tqdm(total=total, file=our_file, miniters=500, mininterval=0.1, maxinterval=maxinterval) as t1:
            with tqdm(total=total, file=our_file, miniters=500, mininterval=0.1, maxinterval=100000.0) as t2:
                cpu_timify(t1, Time)
                cpu_timify(t2, Time)
                Time.fake_sleep(maxinterval / 10)
                t1.update(500)
                t2.update(500)
                assert t1.miniters <= 500
                assert t2.miniters == 500
                Time.fake_sleep(maxinterval)
                t1.update(1)
                t2.update(1)
                timeend = Time.time()
                while not (t1.monitor.woken >= timeend and t1.miniters == 1):
                    Time.fake_sleep(1)
                assert t1.miniters == 1
                assert t2.miniters == 500

def test_imap():
    if False:
        i = 10
        return i + 15
    'Test multiprocessing.Pool'
    try:
        from multiprocessing import Pool
    except ImportError as err:
        skip(str(err))
    pool = Pool()
    res = list(tqdm(pool.imap(incr, range(100)), disable=True))
    pool.close()
    assert res[-1] == 100

@patch_lock(thread=True)
def test_threadpool():
    if False:
        print('Hello World!')
    'Test concurrent.futures.ThreadPoolExecutor'
    ThreadPoolExecutor = importorskip('concurrent.futures').ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        res = list(tqdm(pool.map(incr_bar, range(100)), disable=True))
    assert sum(res) == sum(range(1, 101))
"""Tests for pylab tools module.
"""
import time
from IPython.lib import backgroundjobs as bg
t_short = 0.0001

def sleeper(interval=t_short, *a, **kw):
    if False:
        for i in range(10):
            print('nop')
    args = dict(interval=interval, other_args=a, kw_args=kw)
    time.sleep(interval)
    return args

def crasher(interval=t_short, *a, **kw):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(interval)
    raise Exception('Dead job with interval %s' % interval)

def test_result():
    if False:
        while True:
            i = 10
    'Test job submission and result retrieval'
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper)
    j.join()
    assert j.result['interval'] == t_short

def test_flush():
    if False:
        i = 10
        return i + 15
    'Test job control'
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper)
    j.join()
    assert len(jobs.completed) == 1
    assert len(jobs.dead) == 0
    jobs.flush()
    assert len(jobs.completed) == 0

def test_dead():
    if False:
        for i in range(10):
            print('nop')
    'Test control of dead jobs'
    jobs = bg.BackgroundJobManager()
    j = jobs.new(crasher)
    j.join()
    assert len(jobs.completed) == 0
    assert len(jobs.dead) == 1
    jobs.flush()
    assert len(jobs.dead) == 0

def test_longer():
    if False:
        print('Hello World!')
    'Test control of longer-running jobs'
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper, 0.1)
    assert len(jobs.running) == 1
    assert len(jobs.completed) == 0
    j.join()
    assert len(jobs.running) == 0
    assert len(jobs.completed) == 1
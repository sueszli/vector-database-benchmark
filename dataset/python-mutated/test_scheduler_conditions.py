from rocketry.time import TimeDelta
from rocketry.conditions import SchedulerCycles, SchedulerStarted

def test_scheduler_started(session):
    if False:
        print('Hello World!')
    session.config.shut_cond = ~SchedulerStarted(period=TimeDelta('1 second'))
    session.start()
    assert session.scheduler.n_cycles > 1

def test_scheduler_cycles(session):
    if False:
        for i in range(10):
            print('nop')
    session.config.shut_cond = SchedulerCycles() >= 4
    session.start()
    assert session.scheduler.n_cycles == 4
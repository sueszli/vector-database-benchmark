import time
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.schedulers import timed
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler

class NightlyBase(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):
    """detailed getNextBuildTime tests"""
    OBJECTID = 133
    SCHEDULERID = 33

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.setUpScheduler()

    def makeScheduler(self, firstBuildDuration=0, **kwargs):
        if False:
            print('Hello World!')
        return self.attachScheduler(timed.NightlyBase(**kwargs), self.OBJECTID, self.SCHEDULERID)

    @defer.inlineCallbacks
    def do_getNextBuildTime_test(self, sched, *expectations):
        if False:
            i = 10
            return i + 15
        for (lastActuated, expected) in expectations:
            (lastActuated_ep, expected_ep) = [time.mktime(t + (0,) * (8 - len(t)) + (-1,)) for t in (lastActuated, expected)]
            got_ep = (yield sched.getNextBuildTime(lastActuated_ep))
            self.assertEqual(got_ep, expected_ep, f'{lastActuated} -> {time.localtime(got_ep)} != {expected}')

    def test_getNextBuildTime_hourly(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'])
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 0, 0), (2011, 1, 1, 4, 0, 0)), ((2011, 1, 1, 3, 15, 0), (2011, 1, 1, 4, 0, 0)), ((2011, 1, 1, 3, 15, 1), (2011, 1, 1, 4, 0, 0)), ((2011, 1, 1, 3, 59, 1), (2011, 1, 1, 4, 0, 0)), ((2011, 1, 1, 3, 59, 59), (2011, 1, 1, 4, 0, 0)), ((2011, 1, 1, 23, 22, 22), (2011, 1, 2, 0, 0, 0)), ((2011, 1, 1, 23, 59, 0), (2011, 1, 2, 0, 0, 0)))

    def test_getNextBuildTime_minutes_single(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=4)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 0, 0), (2011, 1, 1, 3, 4, 0)), ((2011, 1, 1, 3, 15, 0), (2011, 1, 1, 4, 4, 0)))

    def test_getNextBuildTime_minutes_multiple(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], minute=[4, 34])
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 0, 0), (2011, 1, 1, 3, 4, 0)), ((2011, 1, 1, 3, 15, 0), (2011, 1, 1, 3, 34, 0)), ((2011, 1, 1, 3, 34, 0), (2011, 1, 1, 4, 4, 0)), ((2011, 1, 1, 3, 59, 1), (2011, 1, 1, 4, 4, 0)))

    def test_getNextBuildTime_minutes_star(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], minute='*')
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 11, 30), (2011, 1, 1, 3, 12, 0)), ((2011, 1, 1, 3, 12, 0), (2011, 1, 1, 3, 13, 0)), ((2011, 1, 1, 3, 59, 0), (2011, 1, 1, 4, 0, 0)))

    def test_getNextBuildTime_hours_single(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], hour=4)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 0), (2011, 1, 1, 4, 0)), ((2011, 1, 1, 13, 0), (2011, 1, 2, 4, 0)))

    def test_getNextBuildTime_hours_multiple(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], hour=[7, 19])
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 0), (2011, 1, 1, 7, 0)), ((2011, 1, 1, 7, 1), (2011, 1, 1, 19, 0)), ((2011, 1, 1, 18, 59), (2011, 1, 1, 19, 0)), ((2011, 1, 1, 19, 59), (2011, 1, 2, 7, 0)))

    def test_getNextBuildTime_hours_minutes(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], hour=13, minute=19)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 1, 3, 11), (2011, 1, 1, 13, 19)), ((2011, 1, 1, 13, 19), (2011, 1, 2, 13, 19)), ((2011, 1, 1, 23, 59), (2011, 1, 2, 13, 19)))

    def test_getNextBuildTime_month_single(self):
        if False:
            return 10
        sched = self.makeScheduler(name='test', builderNames=['test'], month=3)
        return self.do_getNextBuildTime_test(sched, ((2011, 2, 27, 3, 11), (2011, 3, 1, 0, 0)), ((2011, 3, 1, 1, 11), (2011, 3, 1, 2, 0)))

    def test_getNextBuildTime_month_multiple(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], month=[4, 6])
        return self.do_getNextBuildTime_test(sched, ((2011, 3, 30, 3, 11), (2011, 4, 1, 0, 0)), ((2011, 4, 1, 1, 11), (2011, 4, 1, 2, 0)), ((2011, 5, 29, 3, 11), (2011, 6, 1, 0, 0)))

    def test_getNextBuildTime_month_dayOfMonth(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], month=[3, 6], dayOfMonth=[15])
        return self.do_getNextBuildTime_test(sched, ((2011, 2, 12, 3, 11), (2011, 3, 15, 0, 0)), ((2011, 3, 12, 3, 11), (2011, 3, 15, 0, 0)))

    def test_getNextBuildTime_dayOfMonth_single(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfMonth=10)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 9, 3, 0), (2011, 1, 10, 0, 0)), ((2011, 1, 10, 3, 0), (2011, 1, 10, 4, 0)), ((2011, 1, 30, 3, 0), (2011, 2, 10, 0, 0)), ((2011, 12, 30, 11, 0), (2012, 1, 10, 0, 0)))

    def test_getNextBuildTime_dayOfMonth_multiple(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfMonth=[10, 20, 30])
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 9, 22, 0), (2011, 1, 10, 0, 0)), ((2011, 1, 19, 22, 0), (2011, 1, 20, 0, 0)), ((2011, 1, 29, 22, 0), (2011, 1, 30, 0, 0)), ((2011, 2, 29, 22, 0), (2011, 3, 10, 0, 0)))

    def test_getNextBuildTime_dayOfMonth_hours_minutes(self):
        if False:
            i = 10
            return i + 15
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfMonth=15, hour=20, minute=30)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 13, 22, 19), (2011, 1, 15, 20, 30)), ((2011, 1, 15, 19, 19), (2011, 1, 15, 20, 30)), ((2011, 1, 15, 20, 29), (2011, 1, 15, 20, 30)))

    def test_getNextBuildTime_dayOfWeek_single(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfWeek=1)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 3, 22, 19), (2011, 1, 4, 0, 0)), ((2011, 1, 4, 19, 19), (2011, 1, 4, 20, 0)))

    def test_getNextBuildTime_dayOfWeek_single_as_string(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfWeek='1')
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 3, 22, 19), (2011, 1, 4, 0, 0)), ((2011, 1, 4, 19, 19), (2011, 1, 4, 20, 0)))

    def test_getNextBuildTime_dayOfWeek_multiple_as_string(self):
        if False:
            while True:
                i = 10
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfWeek='tue,3')
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 3, 22, 19), (2011, 1, 4, 0, 0)), ((2011, 1, 4, 19, 19), (2011, 1, 4, 20, 0)), ((2011, 1, 5, 22, 19), (2011, 1, 6, 0, 0)), ((2011, 1, 6, 19, 19), (2011, 1, 6, 20, 0)))

    def test_getNextBuildTime_dayOfWeek_multiple_hours(self):
        if False:
            print('Hello World!')
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfWeek=[1, 3], hour=1)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 3, 22, 19), (2011, 1, 4, 1, 0)), ((2011, 1, 4, 22, 19), (2011, 1, 6, 1, 0)))

    def test_getNextBuildTime_dayOfWeek_dayOfMonth(self):
        if False:
            for i in range(10):
                print('nop')
        sched = self.makeScheduler(name='test', builderNames=['test'], dayOfWeek=[1, 4], dayOfMonth=5, hour=1)
        return self.do_getNextBuildTime_test(sched, ((2011, 1, 3, 22, 19), (2011, 1, 4, 1, 0)), ((2011, 1, 4, 22, 19), (2011, 1, 5, 1, 0)), ((2011, 1, 5, 22, 19), (2011, 1, 7, 1, 0)))
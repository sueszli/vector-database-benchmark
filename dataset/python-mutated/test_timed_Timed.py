from twisted.internet import defer
from twisted.internet import task
from twisted.trial import unittest
from buildbot.schedulers import timed
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler

class Timed(scheduler.SchedulerMixin, TestReactorMixin, unittest.TestCase):
    OBJECTID = 928754

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.setUpScheduler()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tearDownScheduler()

    class Subclass(timed.Timed):

        def getNextBuildTime(self, lastActuation):
            if False:
                for i in range(10):
                    print('nop')
            self.got_lastActuation = lastActuation
            return defer.succeed((lastActuation or 1000) + 60)

        def startBuild(self):
            if False:
                print('Hello World!')
            self.started_build = True
            return defer.succeed(None)

    def makeScheduler(self, firstBuildDuration=0, **kwargs):
        if False:
            print('Hello World!')
        sched = self.attachScheduler(self.Subclass(**kwargs), self.OBJECTID)
        self.clock = sched._reactor = task.Clock()
        return sched
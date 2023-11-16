__author__ = 'Serg G. Brester (sebres)'
__copyright__ = 'Copyright (c) 2014 Serg G. Brester'
__license__ = 'GPL'
import os
import sys
import unittest
import tempfile
import time
from ..server.mytime import MyTime
from ..server.ticket import FailTicket, BanTicket
from ..server.failmanager import FailManager
from ..server.observer import Observers, ObserverThread
from ..server.utils import Utils
from .utils import LogCaptureTestCase
from .dummyjail import DummyJail
from .databasetestcase import getFail2BanDb, Fail2BanDb

class BanTimeIncr(LogCaptureTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        'Call before every test case.'
        super(BanTimeIncr, self).setUp()
        self.__jail = DummyJail()
        self.__jail.calcBanTime = self.calcBanTime
        self.Observer = ObserverThread()

    def tearDown(self):
        if False:
            print('Hello World!')
        super(BanTimeIncr, self).tearDown()

    def calcBanTime(self, banTime, banCount):
        if False:
            while True:
                i = 10
        return self.Observer.calcBanTime(self.__jail, banTime, banCount)

    def testDefault(self, multipliers=None):
        if False:
            print('Hello World!')
        a = self.__jail
        a.setBanTimeExtra('increment', 'true')
        self.assertEqual(a.getBanTimeExtra('increment'), True)
        a.setBanTimeExtra('maxtime', '1d')
        self.assertEqual(a.getBanTimeExtra('maxtime'), 24 * 60 * 60)
        a.setBanTimeExtra('rndtime', None)
        a.setBanTimeExtra('factor', None)
        a.setBanTimeExtra('multipliers', multipliers)
        self.assertEqual([a.calcBanTime(600, i) for i in range(1, 11)], [1200, 2400, 4800, 9600, 19200, 38400, 76800, 86400, 86400, 86400])
        a.setBanTimeExtra('maxtime', '30d')
        arr = [1200, 2400, 4800, 9600, 19200, 38400, 76800, 153600, 307200, 614400]
        if multipliers is not None:
            multcnt = len(multipliers.split(' '))
            if multcnt < 11:
                arr = arr[0:multcnt - 1] + [arr[multcnt - 2]] * (11 - multcnt)
        self.assertEqual([a.calcBanTime(600, i) for i in range(1, 11)], arr)
        a.setBanTimeExtra('maxtime', '1d')
        a.setBanTimeExtra('factor', '2')
        self.assertEqual([a.calcBanTime(600, i) for i in range(1, 11)], [2400, 4800, 9600, 19200, 38400, 76800, 86400, 86400, 86400, 86400])
        a.setBanTimeExtra('factor', '1.33')
        self.assertEqual([int(a.calcBanTime(600, i)) for i in range(1, 11)], [1596, 3192, 6384, 12768, 25536, 51072, 86400, 86400, 86400, 86400])
        a.setBanTimeExtra('factor', None)
        a.setBanTimeExtra('maxtime', '12h')
        self.assertEqual([a.calcBanTime(600, i) for i in range(1, 11)], [1200, 2400, 4800, 9600, 19200, 38400, 43200, 43200, 43200, 43200])
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('rndtime', '5m')
        self.assertTrue(False in [1200 in [a.calcBanTime(600, 1) for i in range(10)] for c in range(10)])
        a.setBanTimeExtra('rndtime', None)
        self.assertFalse(False in [1200 in [a.calcBanTime(600, 1) for i in range(10)] for c in range(10)])
        a.setBanTimeExtra('multipliers', None)
        a.setBanTimeExtra('factor', None)
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('rndtime', None)

    def testMultipliers(self):
        if False:
            for i in range(10):
                print('nop')
        self.testDefault('1 2 4 8 16 32 64 128 256')
        self.testDefault(' '.join([str(1 << i) for i in range(31)]))

    def testFormula(self):
        if False:
            print('Hello World!')
        a = self.__jail
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('rndtime', None)
        a.setBanTimeExtra('formula', 'ban.Time * math.exp(float(ban.Count+1)*banFactor)/math.exp(1*banFactor)')
        a.setBanTimeExtra('factor', '2.0 / 2.885385')
        a.setBanTimeExtra('multipliers', None)
        self.assertEqual([int(a.calcBanTime(600, i)) for i in range(1, 11)], [1200, 2400, 4800, 9600, 19200, 38400, 76800, 86400, 86400, 86400])
        a.setBanTimeExtra('maxtime', '30d')
        self.assertEqual([int(a.calcBanTime(600, i)) for i in range(1, 11)], [1200, 2400, 4800, 9600, 19200, 38400, 76800, 153601, 307203, 614407])
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('factor', '1')
        self.assertEqual([int(a.calcBanTime(600, i)) for i in range(1, 11)], [1630, 4433, 12051, 32758, 86400, 86400, 86400, 86400, 86400, 86400])
        a.setBanTimeExtra('factor', '2.0 / 2.885385')
        a.setBanTimeExtra('maxtime', '12h')
        self.assertEqual([int(a.calcBanTime(600, i)) for i in range(1, 11)], [1200, 2400, 4800, 9600, 19200, 38400, 43200, 43200, 43200, 43200])
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('rndtime', '5m')
        self.assertTrue(False in [1200 in [int(a.calcBanTime(600, 1)) for i in range(10)] for c in range(10)])
        a.setBanTimeExtra('rndtime', None)
        self.assertFalse(False in [1200 in [int(a.calcBanTime(600, 1)) for i in range(10)] for c in range(10)])
        a.setBanTimeExtra('factor', None)
        a.setBanTimeExtra('multipliers', None)
        a.setBanTimeExtra('factor', None)
        a.setBanTimeExtra('maxtime', '24h')
        a.setBanTimeExtra('rndtime', None)

class BanTimeIncrDB(LogCaptureTestCase):

    def setUp(self):
        if False:
            return 10
        'Call before every test case.'
        super(BanTimeIncrDB, self).setUp()
        if Fail2BanDb is None:
            raise unittest.SkipTest('Unable to import fail2ban database module as sqlite is not available.')
        elif Fail2BanDb is None:
            return
        (_, self.dbFilename) = tempfile.mkstemp('.db', 'fail2ban_')
        self.db = getFail2BanDb(self.dbFilename)
        self.jail = DummyJail()
        self.jail.database = self.db
        self.Observer = ObserverThread()
        Observers.Main = self.Observer

    def tearDown(self):
        if False:
            print('Hello World!')
        'Call after every test case.'
        if Fail2BanDb is None:
            return
        self.Observer.stop()
        Observers.Main = None
        os.remove(self.dbFilename)
        super(BanTimeIncrDB, self).tearDown()

    def incrBanTime(self, ticket, banTime=None):
        if False:
            i = 10
            return i + 15
        jail = self.jail
        if banTime is None:
            banTime = ticket.getBanTime(jail.actions.getBanTime())
        ticket.setBanTime(None)
        incrTime = self.Observer.incrBanTime(jail, banTime, ticket)
        return incrTime

    def testBanTimeIncr(self):
        if False:
            print('Hello World!')
        if Fail2BanDb is None:
            return
        jail = self.jail
        self.db.addJail(jail)
        jail.actions.setBanTime(10)
        jail.setBanTimeExtra('increment', 'true')
        jail.setBanTimeExtra('multipliers', '1 2 4 8 16 32 64 128 256 512 1024 2048')
        ip = '192.0.2.1'
        stime = int(MyTime.time())
        ticket = FailTicket(ip, stime, [])
        self.assertEqual([self.incrBanTime(ticket, 10) for i in range(3)], [10, 10, 10])
        ticket.incrBanCount()
        self.db.addBan(jail, ticket)
        self.assertEqual([(banCount, timeOfBan, lastBanTime) for (banCount, timeOfBan, lastBanTime) in self.db.getBan(ip, jail, None, False)], [(1, stime, 10)])
        ticket.setTime(stime + 15)
        self.assertEqual(self.incrBanTime(ticket, 10), 20)
        self.db.addBan(jail, ticket)
        self.assertEqual([(banCount, timeOfBan, lastBanTime) for (banCount, timeOfBan, lastBanTime) in self.db.getBan(ip, jail, None, False)], [(2, stime + 15, 20)])
        self.assertEqual([(banCount, timeOfBan, lastBanTime) for (banCount, timeOfBan, lastBanTime) in self.db.getBan(ip, '', None, True)], [(2, stime + 15, 20)])
        self.assertEqual([(banCount, timeOfBan, lastBanTime) for (banCount, timeOfBan, lastBanTime) in self.db.getBan(ip, forbantime=stime, fromtime=stime)], [(2, stime + 15, 20)])
        self.assertEqual(self.db.getCurrentBans(forbantime=-24 * 60 * 60, fromtime=stime, correctBanTime=False), [])
        restored_tickets = self.db.getCurrentBans(ip=ip, correctBanTime=False)
        self.assertEqual(str(restored_tickets), 'FailTicket: ip=%s time=%s bantime=20 bancount=2 #attempts=0 matches=[]' % (ip, stime + 15))
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(str(restored_tickets), '[FailTicket: ip=%s time=%s bantime=20 bancount=2 #attempts=0 matches=[]]' % (ip, stime + 15))
        restored_tickets = self.db.getCurrentBans(jail=jail, fromtime=stime, correctBanTime=False)
        self.assertEqual(str(restored_tickets), '[FailTicket: ip=%s time=%s bantime=20 bancount=2 #attempts=0 matches=[]]' % (ip, stime + 15))
        lastBanTime = 20
        for i in range(10):
            ticket.setTime(stime + lastBanTime + 5)
            banTime = self.incrBanTime(ticket, 10)
            self.assertEqual(banTime, lastBanTime * 2)
            self.db.addBan(jail, ticket)
            lastBanTime = banTime
        ticket.setTime(stime + lastBanTime + 5)
        banTime = self.incrBanTime(ticket, 10)
        self.assertNotEqual(banTime, lastBanTime * 2)
        self.assertEqual(banTime, lastBanTime)
        self.db.addBan(jail, ticket)
        lastBanTime = banTime
        ticket2 = FailTicket(ip + '2', stime - 24 * 60 * 60, [])
        ticket2.setBanTime(12 * 60 * 60)
        ticket2.incrBanCount()
        self.db.addBan(jail, ticket2)
        ticket2 = FailTicket(ip + '1', stime - 24 * 60 * 60, [])
        ticket2.setBanTime(36 * 60 * 60)
        ticket2.incrBanCount()
        self.db.addBan(jail, ticket2)
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 2)
        self.assertEqual(str(restored_tickets[0]), 'FailTicket: ip=%s time=%s bantime=%s bancount=13 #attempts=0 matches=[]' % (ip, stime + lastBanTime + 5, lastBanTime))
        self.assertEqual(str(restored_tickets[1]), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip + '1', stime - 24 * 60 * 60, 36 * 60 * 60))
        restored_tickets = self.db.getCurrentBans(fromtime=stime - 18 * 60 * 60, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 3)
        self.assertEqual(str(restored_tickets[2]), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip + '2', stime - 24 * 60 * 60, 12 * 60 * 60))
        self.assertFalse(restored_tickets[1].isTimedOut(stime))
        self.assertFalse(restored_tickets[1].isTimedOut(stime))
        self.assertTrue(restored_tickets[2].isTimedOut(stime))
        self.assertFalse(restored_tickets[2].isTimedOut(stime - 18 * 60 * 60))
        ticket = FailTicket(ip + '3', stime - 36 * 60 * 60, [])
        self.assertTrue(ticket.isTimedOut(stime, 600))
        self.assertFalse(ticket.isTimedOut(stime, -1))
        ticket.setBanTime(-1)
        self.assertFalse(ticket.isTimedOut(stime, 600))
        self.assertFalse(ticket.isTimedOut(stime, -1))
        ticket.setBanTime(600)
        self.assertTrue(ticket.isTimedOut(stime, -1))
        ticket.setBanTime(-1)
        ticket.incrBanCount()
        self.db.addBan(jail, ticket)
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 3)
        self.assertEqual(str(restored_tickets[2]), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip + '3', stime - 36 * 60 * 60, -1))
        self.db.purge()
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 3)
        ticket.setBanTime(600)
        ticket.incrBanCount()
        self.db.addBan(jail, ticket)
        self.db.purge()
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 2)
        self.assertEqual(restored_tickets[0].getID(), ip)
        self.db._purgeAge = -48 * 60 * 60
        self.db.purge()
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        self.assertEqual(restored_tickets[0].getID(), ip + '1')
        self.db._purgeAge = -240 * 60 * 60
        self.db.purge()
        restored_tickets = self.db.getCurrentBans(fromtime=stime, correctBanTime=False)
        self.assertEqual(restored_tickets, [])
        jail1 = DummyJail(backend='polling')
        jail1.filter.ignoreSelf = False
        jail1.setBanTimeExtra('increment', 'true')
        jail1.database = self.db
        self.db.addJail(jail1)
        jail2 = DummyJail(name='DummyJail-2', backend='polling')
        jail2.filter.ignoreSelf = False
        jail2.database = self.db
        self.db.addJail(jail2)
        ticket1 = FailTicket(ip, stime, [])
        ticket1.setBanTime(6000)
        ticket1.incrBanCount()
        self.db.addBan(jail1, ticket1)
        ticket2 = FailTicket(ip, stime - 6000, [])
        ticket2.setBanTime(12000)
        ticket2.setBanCount(1)
        ticket2.incrBanCount()
        self.db.addBan(jail2, ticket2)
        restored_tickets = self.db.getCurrentBans(jail=jail1, fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        self.assertEqual(str(restored_tickets[0]), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip, stime, 6000))
        restored_tickets = self.db.getCurrentBans(jail=jail2, fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        self.assertEqual(str(restored_tickets[0]), 'FailTicket: ip=%s time=%s bantime=%s bancount=2 #attempts=0 matches=[]' % (ip, stime - 6000, 12000))
        for row in self.db.getBan(ip, jail1):
            self.assertEqual(row, (1, stime, 6000))
            break
        for row in self.db.getBan(ip, jail2):
            self.assertEqual(row, (2, stime - 6000, 12000))
            break
        for row in self.db.getBan(ip, overalljails=True):
            self.assertEqual(row, (3, stime, 18000))
            break
        jail1.restoreCurrentBans(correctBanTime=False)
        ticket = jail1.getFailTicket()
        self.assertTrue(ticket.restored)
        self.assertEqual(str(ticket), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip, stime, 6000))
        jail2.restoreCurrentBans(correctBanTime=False)
        self.assertEqual(jail2.getFailTicket(), False)
        jail1.setBanTimeExtra('maxtime', '10m')
        jail1.restoreCurrentBans()
        ticket = jail1.getFailTicket()
        self.assertTrue(ticket.restored)
        self.assertEqual(str(ticket), 'FailTicket: ip=%s time=%s bantime=%s bancount=1 #attempts=0 matches=[]' % (ip, stime, 600))
        jail2.restoreCurrentBans()
        self.assertEqual(jail2.getFailTicket(), False)

    def testObserver(self):
        if False:
            for i in range(10):
                print('nop')
        if Fail2BanDb is None:
            return
        jail = self.jail = DummyJail(backend='polling')
        jail.database = self.db
        self.db.addJail(jail)
        jail.actions.setBanTime(10)
        jail.setBanTimeExtra('increment', 'true')
        obs = Observers.Main
        obs.start()
        obs.db_set(self.db)
        obs.add('nop')
        obs.wait_empty(5)
        self.db._purgeAge = -240 * 60 * 60
        obs.add_named_timer('DB_PURGE', 0.001, 'db_purge')
        self.assertLogged('Purge database event occurred', wait=True)
        obs.wait_idle(0.025)
        obs.add('nop')
        obs.wait_empty(5)
        stime = int(MyTime.time())
        tickets = self.db.getBans()
        self.assertEqual(tickets, [])
        ip = '192.0.2.1'
        ticket = FailTicket(ip, stime - 120, [])
        failManager = jail.filter.failManager = FailManager()
        failManager.setMaxRetry(3)
        for i in range(3):
            failManager.addFailure(ticket)
            obs.add('failureFound', jail, ticket)
        obs.wait_empty(5)
        self.assertEqual(ticket.getBanCount(), 0)
        self.assertTrue(not jail.getFailTicket())
        ticket.setBanCount(4)
        self.db.addBan(jail, ticket)
        restored_tickets = self.db.getCurrentBans(jail=jail, fromtime=stime - 120, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        ticket = FailTicket(ip, stime, [])
        failManager = jail.filter.failManager = FailManager()
        failManager.setMaxRetry(3)
        failManager.addFailure(ticket)
        obs.add('failureFound', jail, ticket)
        obs.wait_empty(5)
        ticket2 = Utils.wait_for(jail.getFailTicket, 10)
        self.assertTrue(ticket2)
        self.assertEqual(ticket2.getRetry(), failManager.getMaxRetry())
        failticket2 = ticket2
        ticket2 = BanTicket.wrap(failticket2)
        self.assertEqual(ticket2, failticket2)
        obs.add('banFound', ticket2, jail, 10)
        obs.wait_empty(5)
        self.assertEqual(ticket2.getBanTime(), 160)
        self.assertEqual(ticket2.getBanCount(), 5)
        restored_tickets = self.db.getCurrentBans(jail=jail, fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        self.assertEqual(restored_tickets[0].getBanTime(), 160)
        self.assertEqual(restored_tickets[0].getBanCount(), 5)
        ticket = FailTicket(ip, stime - 60, ['test-expired-ban-time'])
        jail.putFailTicket(ticket)
        self.assertFalse(jail.actions.checkBan())
        ticket = FailTicket(ip, MyTime.time(), ['test-actions'])
        jail.putFailTicket(ticket)
        self.assertTrue(jail.actions.checkBan())
        obs.wait_empty(5)
        restored_tickets = self.db.getCurrentBans(jail=jail, fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 1)
        self.assertEqual(restored_tickets[0].getBanTime(), 320)
        self.assertEqual(restored_tickets[0].getBanCount(), 6)
        ticket = FailTicket(ip + '1', MyTime.time(), ['test-permanent'])
        ticket.setBanTime(-1)
        jail.putFailTicket(ticket)
        self.assertTrue(jail.actions.checkBan())
        obs.wait_empty(5)
        ticket = FailTicket(ip + '1', MyTime.time(), ['test-permanent'])
        ticket.setBanTime(600)
        jail.putFailTicket(ticket)
        self.assertFalse(jail.actions.checkBan())
        obs.wait_empty(5)
        restored_tickets = self.db.getCurrentBans(jail=jail, fromtime=stime, correctBanTime=False)
        self.assertEqual(len(restored_tickets), 2)
        self.assertEqual(restored_tickets[1].getBanTime(), -1)
        self.assertEqual(restored_tickets[1].getBanCount(), 1)
        obs.stop()

class ObserverTest(LogCaptureTestCase):

    def setUp(self):
        if False:
            return 10
        'Call before every test case.'
        super(ObserverTest, self).setUp()

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Call after every test case.'
        super(ObserverTest, self).tearDown()

    def testObserverBanTimeIncr(self):
        if False:
            return 10
        obs = ObserverThread()
        obs.start()
        obs.wait_idle(1)
        o = set(['test'])
        obs.add('call', o.clear)
        obs.add('call', o.add, 'test2')
        obs.wait_empty(1)
        self.assertFalse(obs.is_full)
        self.assertEqual(o, set(['test2']))
        obs.paused = True
        obs.add('call', o.clear)
        obs.add('call', o.add, 'test3')
        obs.wait_empty(10 * Utils.DEFAULT_SLEEP_TIME)
        self.assertTrue(obs.is_full)
        self.assertEqual(o, set(['test2']))
        obs.paused = False
        obs.wait_empty(1)
        self.assertEqual(o, set(['test3']))
        self.assertTrue(obs.isActive())
        self.assertTrue(obs.isAlive())
        obs.stop()
        obs = None

    class _BadObserver(ObserverThread):

        def run(self):
            if False:
                while True:
                    i = 10
            raise RuntimeError('run bad thread exception')

    def testObserverBadRun(self):
        if False:
            for i in range(10):
                print('nop')
        obs = ObserverTest._BadObserver()
        obs.wait_empty = lambda v: ()
        prev_exchook = sys.__excepthook__
        x = []
        sys.__excepthook__ = lambda *args: x.append(args)
        try:
            obs.start()
            obs.stop()
            obs.join()
            self.assertTrue(Utils.wait_for(lambda : len(x) and self._is_logged('Unhandled exception'), 3))
        finally:
            sys.__excepthook__ = prev_exchook
        self.assertLogged('Unhandled exception')
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0][0], RuntimeError)
        self.assertEqual(str(x[0][1]), 'run bad thread exception')
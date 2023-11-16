__copyright__ = 'Copyright (c) 2013 Steven Hiscocks'
__license__ = 'GPL'
import os
import sys
import unittest
import tempfile
import sqlite3
import shutil
from ..server.filter import FileContainer, Filter
from ..server.mytime import MyTime
from ..server.ticket import FailTicket
from ..server.actions import Actions, Utils
from .dummyjail import DummyJail
try:
    from ..server import database
    Fail2BanDb = database.Fail2BanDb
except ImportError:
    Fail2BanDb = None
from .utils import LogCaptureTestCase, logSys as DefLogSys
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')

def getFail2BanDb(filename):
    if False:
        print('Hello World!')
    if unittest.F2B.memory_db:
        return Fail2BanDb(':memory:')
    return Fail2BanDb(filename)

class DatabaseTest(LogCaptureTestCase):

    def setUp(self):
        if False:
            return 10
        'Call before every test case.'
        super(DatabaseTest, self).setUp()
        if Fail2BanDb is None:
            raise unittest.SkipTest('Unable to import fail2ban database module as sqlite is not available.')
        self.dbFilename = None
        if not unittest.F2B.memory_db:
            (_, self.dbFilename) = tempfile.mkstemp('.db', 'fail2ban_')
        self._db = ':auto-create-in-memory:'

    @property
    def db(self):
        if False:
            print('Hello World!')
        if isinstance(self._db, str) and self._db == ':auto-create-in-memory:':
            self._db = getFail2BanDb(self.dbFilename)
        return self._db

    @db.setter
    def db(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(self._db, Fail2BanDb):
            self._db.close()
        self._db = value

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Call after every test case.'
        super(DatabaseTest, self).tearDown()
        if Fail2BanDb is None:
            return
        if self.dbFilename is not None:
            os.remove(self.dbFilename)

    def testGetFilename(self):
        if False:
            for i in range(10):
                print('nop')
        if self.db.filename == ':memory:':
            raise unittest.SkipTest('in :memory: database')
        self.assertEqual(self.dbFilename, self.db.filename)

    def testPurgeAge(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.db.purgeage, 86400)
        self.db.purgeage = '1y6mon15d5h30m'
        self.assertEqual(self.db.purgeage, 48652200)
        self.db.purgeage = '2y 12mon 30d 10h 60m'
        self.assertEqual(self.db.purgeage, 48652200 * 2)

    def testCreateInvalidPath(self):
        if False:
            print('Hello World!')
        self.assertRaises(sqlite3.OperationalError, Fail2BanDb, '/this/path/should/not/exist')

    def testCreateAndReconnect(self):
        if False:
            i = 10
            return i + 15
        if self.db.filename == ':memory:':
            raise unittest.SkipTest('in :memory: database')
        self.testAddJail()
        self.db = Fail2BanDb(self.dbFilename)
        self.assertTrue(self.jail.name in self.db.getJailNames(), 'Jail not retained in Db after disconnect reconnect.')

    def testRepairDb(self):
        if False:
            i = 10
            return i + 15
        if not Utils.executeCmd('sqlite3 --version'):
            raise unittest.SkipTest('no sqlite3 command')
        self.db = None
        if self.dbFilename is None:
            (_, self.dbFilename) = tempfile.mkstemp('.db', 'fail2ban_')
        for truncSize in (14000, 4000):
            self.pruneLog('[test-repair], next phase - file-size: %d' % truncSize)
            shutil.copyfile(os.path.join(TEST_FILES_DIR, 'database_v1.db'), self.dbFilename)
            f = os.open(self.dbFilename, os.O_RDWR)
            os.ftruncate(f, truncSize)
            os.close(f)
            try:
                self.db = Fail2BanDb(self.dbFilename)
                if truncSize == 14000:
                    self.assertLogged('Repair seems to be successful', 'Check integrity', 'Database updated', all=True)
                    self.assertEqual(self.db.getLogPaths(), set(['/tmp/Fail2BanDb_pUlZJh.log']))
                    self.assertEqual(len(self.db.getJailNames()), 1)
                else:
                    self.assertLogged('Repair seems to be failed', 'Check integrity', 'New database created.', all=True)
                    self.assertEqual(len(self.db.getLogPaths()), 0)
                    self.assertEqual(len(self.db.getJailNames()), 0)
            finally:
                if self.db and self.db._dbFilename != ':memory:':
                    os.remove(self.db._dbBackupFilename)
                    self.db = None

    def testUpdateDb(self):
        if False:
            while True:
                i = 10
        self.db = None
        try:
            if self.dbFilename is None:
                (_, self.dbFilename) = tempfile.mkstemp('.db', 'fail2ban_')
            shutil.copyfile(os.path.join(TEST_FILES_DIR, 'database_v1.db'), self.dbFilename)
            self.db = Fail2BanDb(self.dbFilename)
            self.assertEqual(self.db.getJailNames(), set(['DummyJail #29162448 with 0 tickets']))
            self.assertEqual(self.db.getLogPaths(), set(['/tmp/Fail2BanDb_pUlZJh.log']))
            ticket = FailTicket('127.0.0.1', 1388009242.26, ['abc\n'])
            self.assertEqual(self.db.getBans()[0], ticket)
            self.assertEqual(self.db.updateDb(Fail2BanDb.__version__), Fail2BanDb.__version__)
            self.assertRaises(NotImplementedError, self.db.updateDb, Fail2BanDb.__version__ + 1)
            tickets = self.db.getCurrentBans(fromtime=1388009242, correctBanTime=123456)
            self.assertEqual(len(tickets), 1)
            self.assertEqual(tickets[0].getBanTime(), 123456)
        finally:
            if self.db and self.db._dbFilename != ':memory:':
                os.remove(self.db._dbBackupFilename)

    def testUpdateDb2(self):
        if False:
            i = 10
            return i + 15
        self.db = None
        if self.dbFilename is None:
            (_, self.dbFilename) = tempfile.mkstemp('.db', 'fail2ban_')
        shutil.copyfile(os.path.join(TEST_FILES_DIR, 'database_v2.db'), self.dbFilename)
        self.db = Fail2BanDb(self.dbFilename)
        self.assertEqual(self.db.getJailNames(), set(['pam-generic']))
        self.assertEqual(self.db.getLogPaths(), set(['/var/log/auth.log']))
        bans = self.db.getBans()
        self.assertEqual(len(bans), 2)
        ticket = FailTicket('1.2.3.7', 1417595494, ['Dec  3 09:31:08 f2btest test:auth[27658]: pam_unix(test:auth): authentication failure; logname= uid=0 euid=0 tty=test ruser= rhost=1.2.3.7', 'Dec  3 09:31:32 f2btest test:auth[27671]: pam_unix(test:auth): authentication failure; logname= uid=0 euid=0 tty=test ruser= rhost=1.2.3.7', 'Dec  3 09:31:34 f2btest test:auth[27673]: pam_unix(test:auth): authentication failure; logname= uid=0 euid=0 tty=test ruser= rhost=1.2.3.7'])
        ticket.setAttempt(3)
        self.assertEqual(bans[0], ticket)
        self.assertEqual(bans[1].getID(), '1.2.3.8')
        self.assertEqual(self.db.updateDb(Fail2BanDb.__version__), Fail2BanDb.__version__)
        self.jail = DummyJail(name='pam-generic')
        tickets = self.db.getCurrentBans(jail=self.jail, fromtime=1417595494)
        self.assertEqual(len(tickets), 2)
        self.assertEqual(tickets[0].getBanTime(), 600)
        self.assertRaises(NotImplementedError, self.db.updateDb, Fail2BanDb.__version__ + 1)
        os.remove(self.db._dbBackupFilename)

    def testAddJail(self):
        if False:
            i = 10
            return i + 15
        self.jail = DummyJail()
        self.db.addJail(self.jail)
        self.assertTrue(self.jail.name in self.db.getJailNames(True), 'Jail not added to database')

    def _testAddLog(self):
        if False:
            return 10
        self.testAddJail()
        (_, filename) = tempfile.mkstemp('.log', 'Fail2BanDb_')
        self.fileContainer = FileContainer(filename, 'utf-8')
        pos = self.db.addLog(self.jail, self.fileContainer)
        self.assertTrue(pos is None)
        self.assertIn(filename, self.db.getLogPaths(self.jail))
        os.remove(filename)

    def testUpdateLog(self):
        if False:
            while True:
                i = 10
        self._testAddLog()
        filename = self.fileContainer.getFileName()
        file_ = open(filename, 'w')
        file_.write('Some text to write which will change md5sum\n')
        file_.close()
        self.fileContainer.open()
        self.fileContainer.readline()
        self.fileContainer.close()
        lastPos = self.fileContainer.getPos()
        self.assertTrue(lastPos > 0)
        self.db.updateLog(self.jail, self.fileContainer)
        self.fileContainer = FileContainer(filename, 'utf-8')
        self.assertEqual(self.fileContainer.getPos(), 0)
        self.assertEqual(self.db.addLog(self.jail, self.fileContainer), lastPos)
        file_ = open(filename, 'w')
        file_.write('Some different text to change md5sum\n')
        file_.close()
        self.fileContainer = FileContainer(filename, 'utf-8')
        self.assertEqual(self.fileContainer.getPos(), 0)
        self.assertEqual(self.db.addLog(self.jail, self.fileContainer), None)
        os.remove(filename)

    def testUpdateJournal(self):
        if False:
            while True:
                i = 10
        self.testAddJail()
        self.assertEqual(self.db.getJournalPos(self.jail, 'systemd-journal'), None)
        for t in (1500000000, 1500000001, 1500000002):
            self.db.updateJournal(self.jail, 'systemd-journal', t, 'TEST' + str(t))
            self.assertEqual(self.db.getJournalPos(self.jail, 'systemd-journal'), t)

    def testAddBan(self):
        if False:
            return 10
        self.testAddJail()
        ticket = FailTicket('127.0.0.1', 0, ['abc\n'])
        self.db.addBan(self.jail, ticket)
        tickets = self.db.getBans(jail=self.jail)
        self.assertEqual(len(tickets), 1)
        self.assertTrue(isinstance(tickets[0], FailTicket))

    def testAddBanInvalidEncoded(self):
        if False:
            for i in range(10):
                print('nop')
        self.testAddJail()
        tickets = [FailTicket('127.0.0.1', 0, ['user "test"', 'user "Ñâåòà"', 'user "Ã¤Ã¶Ã¼Ã\x9f"']), FailTicket('127.0.0.2', 0, ['user "test"', 'user "Ñâåòà"', 'user "Ã¤Ã¶Ã¼Ã\x9f"']), FailTicket('127.0.0.3', 0, ['user "test"', b'user "\xd1\xe2\xe5\xf2\xe0"', b'user "\xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f"']), FailTicket('127.0.0.4', 0, ['user "test"', 'user "Ñâåòà"', 'user "äöüß"']), FailTicket('127.0.0.5', 0, ['user "test"', 'unterminated Ï']), FailTicket('127.0.0.6', 0, ['user "test"', 'unterminated Ï']), FailTicket('127.0.0.7', 0, ['user "test"', b'unterminated \xcf'])]
        for ticket in tickets:
            self.db.addBan(self.jail, ticket)
        self.assertNotLogged('json dumps failed')
        readtickets = self.db.getBans(jail=self.jail)
        self.assertNotLogged('json loads failed')
        self.assertEqual(len(readtickets), 7)
        for (i, ticket) in enumerate(tickets):
            DefLogSys.debug('readtickets[%d]: %r', i, readtickets[i].getData())
            DefLogSys.debug(' == tickets[%d]: %r', i, ticket.getData())
            self.assertEqual(readtickets[i].getID(), ticket.getID())
            self.assertEqual(len(readtickets[i].getMatches()), len(ticket.getMatches()))
        self.pruneLog('[test-phase 2] simulate errors')
        priorEnc = database.PREFER_ENC
        try:
            database.PREFER_ENC = 'f2b-test::non-existing-encoding'
            for ticket in tickets:
                self.db.addBan(self.jail, ticket)
            self.assertLogged('json dumps failed')
            readtickets = self.db.getBans(jail=self.jail)
            self.assertLogged('json loads failed')
            self.assertEqual(len(readtickets), 14)
        finally:
            database.PREFER_ENC = priorEnc
        self.pruneLog('[test-phase 3] still operable?')
        self.db.addBan(self.jail, FailTicket('127.0.0.8'))
        readtickets = self.db.getBans(jail=self.jail)
        self.assertEqual(len(readtickets), 15)
        self.assertNotLogged('json loads failed', 'json dumps failed')

    def _testAdd3Bans(self):
        if False:
            while True:
                i = 10
        self.testAddJail()
        for i in (1, 2, 3):
            ticket = FailTicket('192.0.2.%d' % i, 0, ['test\n'])
            self.db.addBan(self.jail, ticket)
        tickets = self.db.getBans(jail=self.jail)
        self.assertEqual(len(tickets), 3)
        return tickets

    def testDelBan(self):
        if False:
            for i in range(10):
                print('nop')
        tickets = self._testAdd3Bans()
        self.db.delBan(self.jail, tickets[0].getID())
        self.assertEqual(len(self.db.getBans(jail=self.jail)), 2)
        self.db.delBan(self.jail, tickets[1].getID(), tickets[2].getID())
        self.assertEqual(len(self.db.getBans(jail=self.jail)), 0)

    def testFlushBans(self):
        if False:
            while True:
                i = 10
        self._testAdd3Bans()
        self.db.delBan(self.jail)
        self.assertEqual(len(self.db.getBans(jail=self.jail)), 0)

    def testGetBansWithTime(self):
        if False:
            for i in range(10):
                print('nop')
        self.testAddJail()
        self.db.addBan(self.jail, FailTicket('127.0.0.1', MyTime.time() - 60, ['abc\n']))
        self.db.addBan(self.jail, FailTicket('127.0.0.1', MyTime.time() - 40, ['abc\n']))
        self.assertEqual(len(self.db.getBans(jail=self.jail, bantime=50)), 1)
        self.assertEqual(len(self.db.getBans(jail=self.jail, bantime=20)), 0)
        self.assertEqual(len(self.db.getBans(jail=self.jail, bantime=-1)), 2)

    def testGetBansMerged_MaxMatches(self):
        if False:
            for i in range(10):
                print('nop')
        self.testAddJail()
        maxMatches = 2
        failures = [{'matches': ['abc\n'], 'user': set(['test'])}, {'matches': ['123\n'], 'user': set(['test'])}, {'matches': ['ABC\n'], 'user': set(['test', 'root'])}, {'matches': ['1234\n'], 'user': set(['test', 'root'])}]
        matches2find = [f['matches'][0] for f in failures]
        i = 80
        for f in failures:
            i -= 10
            ticket = FailTicket('127.0.0.1', MyTime.time() - i, data=f)
            ticket.setAttempt(1)
            self.db.addBan(self.jail, ticket)
        self.db.maxMatches = maxMatches
        ticket = self.db.getBansMerged('127.0.0.1')
        self.assertEqual(ticket.getID(), '127.0.0.1')
        self.assertEqual(ticket.getAttempt(), len(failures))
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), matches2find[-maxMatches:])
        ticket = FailTicket('127.0.0.1', MyTime.time() - 10, matches2find, data={'user': set(['test', 'root'])})
        ticket.setAttempt(len(failures))
        self.db.addBan(self.jail, ticket)
        ticket = self.db.getBansMerged('127.0.0.1')
        self.assertEqual(ticket.getAttempt(), 2 * len(failures))
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), matches2find[-maxMatches:])
        ticket = self.db.getCurrentBans(self.jail, '127.0.0.1', fromtime=MyTime.time() - 100)
        self.assertTrue(ticket is not None)
        self.assertEqual(ticket.getAttempt(), len(failures))
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), matches2find[-maxMatches:])
        ticket = self.db.getCurrentBans(self.jail, '127.0.0.1', fromtime=MyTime.time() - 100, maxmatches=1)
        self.assertEqual(len(ticket.getMatches()), 1)
        self.assertEqual(ticket.getMatches(), failures[3]['matches'])
        ticket = self.db.getCurrentBans(self.jail, '127.0.0.1', fromtime=MyTime.time() - 100, maxmatches=0)
        self.assertEqual(len(ticket.getMatches()), 0)
        ticket.setMatches(['1', '2', '3'])
        self.db.maxMatches = 0
        self.db.addBan(self.jail, ticket)
        ticket = self.db.getCurrentBans(self.jail, '127.0.0.1', fromtime=MyTime.time() - 100)
        self.assertTrue(ticket is not None)
        self.assertEqual(ticket.getAttempt(), len(failures))
        self.assertEqual(len(ticket.getMatches()), 0)

    def testGetBansMerged(self):
        if False:
            while True:
                i = 10
        self.testAddJail()
        jail2 = DummyJail(name='DummyJail-2')
        self.db.addJail(jail2)
        ticket = FailTicket('127.0.0.1', MyTime.time() - 40, ['abc\n'])
        ticket.setAttempt(10)
        self.db.addBan(self.jail, ticket)
        ticket = FailTicket('127.0.0.1', MyTime.time() - 30, ['123\n'])
        ticket.setAttempt(20)
        self.db.addBan(self.jail, ticket)
        ticket = FailTicket('127.0.0.2', MyTime.time() - 20, ['ABC\n'])
        ticket.setAttempt(30)
        self.db.addBan(self.jail, ticket)
        ticket = FailTicket('127.0.0.1', MyTime.time() - 10, ['ABC\n'])
        ticket.setAttempt(40)
        self.db.addBan(jail2, ticket)
        ticket = self.db.getBansMerged('127.0.0.1')
        self.assertEqual(ticket.getID(), '127.0.0.1')
        self.assertEqual(ticket.getAttempt(), 70)
        self.assertEqual(ticket.getMatches(), ['abc\n', '123\n', 'ABC\n'])
        ticket = self.db.getBansMerged('127.0.0.1', jail=self.jail)
        self.assertEqual(ticket.getID(), '127.0.0.1')
        self.assertEqual(ticket.getAttempt(), 30)
        self.assertEqual(ticket.getMatches(), ['abc\n', '123\n'])
        self.assertEqual(id(ticket), id(self.db.getBansMerged('127.0.0.1', jail=self.jail)))
        newTicket = FailTicket('127.0.0.2', MyTime.time() - 20, ['ABC\n'])
        ticket.setAttempt(40)
        self.db.addBan(self.jail, newTicket)
        self.assertEqual(id(ticket), id(self.db.getBansMerged('127.0.0.1', jail=self.jail)))
        newTicket = FailTicket('127.0.0.1', MyTime.time() - 10, ['ABC\n'])
        ticket.setAttempt(40)
        self.db.addBan(self.jail, newTicket)
        self.assertNotEqual(id(ticket), id(self.db.getBansMerged('127.0.0.1', jail=self.jail)))
        tickets = self.db.getBansMerged()
        self.assertEqual(len(tickets), 2)
        self.assertSortedEqual(list(set((ticket.getID() for ticket in tickets))), [ticket.getID() for ticket in tickets])
        tickets = self.db.getBansMerged(jail=jail2)
        self.assertEqual(len(tickets), 1)
        tickets = self.db.getBansMerged(bantime=25)
        self.assertEqual(len(tickets), 2)
        tickets = self.db.getBansMerged(bantime=15)
        self.assertEqual(len(tickets), 1)
        tickets = self.db.getBansMerged(bantime=5)
        self.assertEqual(len(tickets), 0)
        tickets = self.db.getBansMerged(bantime=-1)
        self.assertEqual(len(tickets), 2)
        tickets = self.db.getCurrentBans(jail=self.jail)
        self.assertEqual(len(tickets), 2)
        ticket = self.db.getCurrentBans(jail=None, ip='127.0.0.1')
        self.assertEqual(ticket.getID(), '127.0.0.1')
        tickets = self.db.getCurrentBans(jail=self.jail, forbantime=15, fromtime=MyTime.time())
        self.assertEqual(len(tickets), 1)
        tickets = self.db.getCurrentBans(jail=self.jail, forbantime=15, fromtime=MyTime.time() + MyTime.str2seconds('1year'))
        self.assertEqual(len(tickets), 0)
        tickets = self.db.getCurrentBans(jail=self.jail, forbantime=-1, fromtime=MyTime.time() + MyTime.str2seconds('1year'))
        self.assertEqual(len(tickets), 0)
        ticket.setBanTime(-1)
        self.db.addBan(self.jail, ticket)
        tickets = self.db.getCurrentBans(jail=self.jail, forbantime=-1, fromtime=MyTime.time() + MyTime.str2seconds('1year'))
        self.assertEqual(len(tickets), 0)
        self.assertLogged('ignore ticket (with new max ban-time %r)' % self.jail.getMaxBanTime())
        self.jail.actions.setBanTime(-1)
        tickets = self.db.getCurrentBans(jail=self.jail, forbantime=-1, fromtime=MyTime.time() + MyTime.str2seconds('1year'))
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0].getBanTime(), -1)

    def testActionWithDB(self):
        if False:
            for i in range(10):
                print('nop')
        self.testAddJail()
        self.jail.database = self.db
        self.db.addJail(self.jail)
        actions = self.jail.actions
        actions.add('action_checkainfo', os.path.join(TEST_FILES_DIR, 'action.d/action_checkainfo.py'), {})
        actions.banManager.setBanTotal(20)
        self.jail._Jail__filter = flt = Filter(self.jail)
        flt.failManager.setFailTotal(50)
        ticket = FailTicket('1.2.3.4')
        ticket.setAttempt(5)
        ticket.setMatches(['test', 'test'])
        self.jail.putFailTicket(ticket)
        actions._Actions__checkBan()
        self.assertLogged('ban ainfo %s, %s, %s, %s' % (True, True, True, True))
        self.assertLogged('jail info %d, %d, %d, %d' % (1, 21, 0, 50))

    def testDelAndAddJail(self):
        if False:
            return 10
        self.testAddJail()
        self.db.delJail(self.jail)
        jails = self.db.getJailNames()
        self.assertIn(len(jails) == 1 and self.jail.name, jails)
        jails = self.db.getJailNames(enabled=False)
        self.assertIn(len(jails) == 1 and self.jail.name, jails)
        jails = self.db.getJailNames(enabled=True)
        self.assertTrue(len(jails) == 0)
        self.db.addJail(self.jail)
        jails = self.db.getJailNames()
        self.assertIn(len(jails) == 1 and self.jail.name, jails)
        jails = self.db.getJailNames(enabled=True)
        self.assertIn(len(jails) == 1 and self.jail.name, jails)
        jails = self.db.getJailNames(enabled=False)
        self.assertTrue(len(jails) == 0)

    def testPurge(self):
        if False:
            while True:
                i = 10
        self.testAddJail()
        self.db.purge()
        self.assertEqual(len(self.db.getJailNames()), 1)
        self.db.delJail(self.jail)
        self.db.purge()
        self.assertEqual(len(self.db.getJailNames()), 0)
        self.testAddBan()
        self.db.delJail(self.jail)
        self.db.purge()
        self.assertEqual(len(self.db.getJailNames()), 0)
        self.assertEqual(len(self.db.getBans(jail=self.jail)), 0)
        self.testAddJail()
        self.db.addBan(self.jail, FailTicket('127.0.0.1', MyTime.time(), ['abc\n']))
        self.db.delJail(self.jail)
        self.db.purge()
        self.assertEqual(len(self.db.getJailNames()), 1)
        self.assertEqual(len(self.db.getBans(jail=self.jail)), 1)
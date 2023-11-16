__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import unittest
from ..server import failmanager
from ..server.failmanager import FailManager, FailManagerEmpty
from ..server.ipdns import IPAddr
from ..server.ticket import FailTicket

class AddFailure(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Call before every test case.'
        super(AddFailure, self).setUp()
        self.__items = None
        self.__failManager = FailManager()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Call after every test case.'
        super(AddFailure, self).tearDown()

    def _addDefItems(self):
        if False:
            while True:
                i = 10
        self.__items = [['193.168.0.128', 1167605999.0], ['193.168.0.128', 1167605999.0], ['193.168.0.128', 1167605999.0], ['193.168.0.128', 1167605999.0], ['193.168.0.128', 1167605999.0], ['87.142.124.10', 1167605999.0], ['87.142.124.10', 1167605999.0], ['87.142.124.10', 1167605999.0], ['100.100.10.10', 1000000000.0], ['100.100.10.10', 1000000500.0], ['100.100.10.10', 1000001000.0], ['100.100.10.10', 1000001500.0], ['100.100.10.10', 1000002000.0]]
        for i in self.__items:
            self.__failManager.addFailure(FailTicket(i[0], i[1]))

    def testFailManagerAdd(self):
        if False:
            while True:
                i = 10
        self._addDefItems()
        self.assertEqual(self.__failManager.size(), 3)
        self.assertEqual(self.__failManager.getFailTotal(), 13)
        self.__failManager.setFailTotal(0)
        self.assertEqual(self.__failManager.getFailTotal(), 0)
        self.__failManager.setFailTotal(13)

    def testFailManagerAdd_MaxMatches(self):
        if False:
            for i in range(10):
                print('nop')
        maxMatches = 2
        self.__failManager.maxMatches = maxMatches
        failures = ['abc\n', '123\n', 'ABC\n', '1234\n']
        i = 80
        for f in failures:
            i -= 10
            ticket = FailTicket('127.0.0.1', 1000002000 - i, [f])
            ticket.setAttempt(1)
            self.__failManager.addFailure(ticket)
        manFailList = self.__failManager._FailManager__failList
        self.assertEqual(len(manFailList), 1)
        ticket = manFailList['127.0.0.1']
        self.assertEqual(ticket.getAttempt(), len(failures))
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), failures[len(failures) - maxMatches:])
        ticket = FailTicket('127.0.0.1', 1000002000 - 10, failures)
        ticket.setAttempt(len(failures))
        self.__failManager.addFailure(ticket)
        manFailList = self.__failManager._FailManager__failList
        self.assertEqual(len(manFailList), 1)
        ticket = manFailList['127.0.0.1']
        self.assertEqual(ticket.getAttempt(), 2 * len(failures))
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), failures[len(failures) - maxMatches:])
        self.__failManager.addFailure(ticket)
        manFailList = self.__failManager._FailManager__failList
        self.assertEqual(len(manFailList), 1)
        ticket = manFailList['127.0.0.1']
        self.assertEqual(ticket.getAttempt(), 2 * len(failures) + 1)
        self.assertEqual(len(ticket.getMatches()), maxMatches)
        self.assertEqual(ticket.getMatches(), failures[len(failures) - maxMatches:])
        self.__failManager.maxMatches = 0
        self.__failManager.addFailure(ticket)
        manFailList = self.__failManager._FailManager__failList
        ticket = manFailList['127.0.0.1']
        self.assertEqual(len(ticket.getMatches()), 0)
        ticket.setMatches(None)

    def testFailManagerMaxTime(self):
        if False:
            while True:
                i = 10
        self._addDefItems()
        self.assertEqual(self.__failManager.getMaxTime(), 600)
        self.__failManager.setMaxTime(13)
        self.assertEqual(self.__failManager.getMaxTime(), 13)
        self.__failManager.setMaxTime(600)

    def testDel(self):
        if False:
            print('Hello World!')
        self._addDefItems()
        self.__failManager.delFailure('193.168.0.128')
        self.__failManager.delFailure('111.111.1.111')
        self.assertEqual(self.__failManager.size(), 2)

    def testCleanupOK(self):
        if False:
            return 10
        self._addDefItems()
        timestamp = 1167606999.0
        self.__failManager.cleanup(timestamp)
        self.assertEqual(self.__failManager.size(), 0)

    def testCleanupNOK(self):
        if False:
            i = 10
            return i + 15
        self._addDefItems()
        timestamp = 1167605990.0
        self.__failManager.cleanup(timestamp)
        self.assertEqual(self.__failManager.size(), 2)

    def testbanOK(self):
        if False:
            for i in range(10):
                print('nop')
        self._addDefItems()
        self.__failManager.setMaxRetry(5)
        ticket = self.__failManager.toBan()
        self.assertEqual(ticket.getID(), '193.168.0.128')
        self.assertTrue(isinstance(ticket.getID(), (str, IPAddr)))
        ticket_str = str(ticket)
        ticket_repr = repr(ticket)
        self.assertEqual(ticket_str, 'FailTicket: ip=193.168.0.128 time=1167605999.0 bantime=None bancount=0 #attempts=5 matches=[]')
        self.assertEqual(ticket_repr, 'FailTicket: ip=193.168.0.128 time=1167605999.0 bantime=None bancount=0 #attempts=5 matches=[]')
        self.assertFalse(not ticket)
        ticket.setTime(1000002000.0)
        self.assertEqual(ticket.getTime(), 1000002000.0)
        self.assertEqual(str(ticket), 'FailTicket: ip=193.168.0.128 time=1000002000.0 bantime=None bancount=0 #attempts=5 matches=[]')

    def testbanNOK(self):
        if False:
            i = 10
            return i + 15
        self._addDefItems()
        self.__failManager.setMaxRetry(10)
        self.assertRaises(FailManagerEmpty, self.__failManager.toBan)

    def testWindow(self):
        if False:
            return 10
        self._addDefItems()
        ticket = self.__failManager.toBan()
        self.assertNotEqual(ticket.getID(), '100.100.10.10')
        ticket = self.__failManager.toBan()
        self.assertNotEqual(ticket.getID(), '100.100.10.10')
        self.assertRaises(FailManagerEmpty, self.__failManager.toBan)

    def testBgService(self):
        if False:
            print('Hello World!')
        bgSvc = self.__failManager._FailManager__bgSvc
        failManager2nd = FailManager()
        bgSvc2 = failManager2nd._FailManager__bgSvc
        self.assertTrue(id(bgSvc) == id(bgSvc2))
        bgSvc2 = None
        self.assertTrue(bgSvc.service(True, True))
        self.assertFalse(bgSvc.service())
        for i in range(1, bgSvc._BgService__threshold):
            self.assertFalse(bgSvc.service())
        bgSvc._BgService__serviceTime = -2147483647
        self.assertTrue(bgSvc.service())
        bgSvc._BgService__serviceTime = -2147483647
        for i in range(1, bgSvc._BgService__threshold):
            self.assertFalse(bgSvc.service())
        self.assertTrue(bgSvc.service(False, True))
        self.assertFalse(bgSvc.service(False, True))

class FailmanagerComplex(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        'Call before every test case.'
        super(FailmanagerComplex, self).setUp()
        self.__failManager = FailManager()
        self.__saved_ll = failmanager.logLevel
        failmanager.logLevel = 3

    def tearDown(self):
        if False:
            while True:
                i = 10
        super(FailmanagerComplex, self).tearDown()
        failmanager.logLevel = self.__saved_ll

    @staticmethod
    def _ip_range(maxips):
        if False:
            for i in range(10):
                print('nop')

        class _ip(list):

            def __str__(self):
                if False:
                    while True:
                        i = 10
                return '.'.join(map(str, self))

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return str(self)

            def __key__(self):
                if False:
                    print('Hello World!')
                return str(self)

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return int(self[0] << 24 | self[1] << 16 | self[2] << 8 | self[3])
        i = 0
        c = [127, 0, 0, 0]
        while i < maxips:
            for n in range(3, 0, -1):
                if c[n] < 255:
                    c[n] += 1
                    break
                c[n] = 0
            yield (i, _ip(c))
            i += 1

    def testCheckIPGenerator(self):
        if False:
            i = 10
            return i + 15
        for (i, ip) in self._ip_range(65536 if not unittest.F2B.fast else 1000):
            if i == 254:
                self.assertEqual(str(ip), '127.0.0.255')
            elif i == 255:
                self.assertEqual(str(ip), '127.0.1.0')
            elif i == 1000:
                self.assertEqual(str(ip), '127.0.3.233')
            elif i == 65534:
                self.assertEqual(str(ip), '127.0.255.255')
            elif i == 65535:
                self.assertEqual(str(ip), '127.1.0.0')
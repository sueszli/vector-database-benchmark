__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import unittest
from .utils import setUpMyTime, tearDownMyTime
from ..server.banmanager import BanManager
from ..server.ipdns import DNSUtils
from ..server.ticket import BanTicket

class AddFailure(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Call before every test case.'
        super(AddFailure, self).setUp()
        setUpMyTime()
        self.__ticket = BanTicket('193.168.0.128', 1167605999.0)
        self.__banManager = BanManager()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Call after every test case.'
        super(AddFailure, self).tearDown()
        tearDownMyTime()

    def testAdd(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        self.assertEqual(self.__banManager.size(), 1)
        self.assertEqual(self.__banManager.getBanTotal(), 1)
        self.__banManager.setBanTotal(0)
        self.assertEqual(self.__banManager.getBanTotal(), 0)

    def testAddDuplicate(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        self.assertFalse(self.__banManager.addBanTicket(self.__ticket))
        self.assertEqual(self.__banManager.size(), 1)

    def testAddDuplicateWithTime(self):
        if False:
            print('Hello World!')
        defBanTime = self.__banManager.getBanTime()
        prevEndOfBanTime = 0
        for (tnew, btnew) in ((1167605999.0, None), (1167605999.0 + 100, None), (1167605999.0, 24 * 60 * 60), (1167605999.0, -1)):
            ticket1 = BanTicket('193.168.0.128', 1167605999.0)
            ticket2 = BanTicket('193.168.0.128', tnew)
            if btnew is not None:
                ticket2.setBanTime(btnew)
            self.assertTrue(self.__banManager.addBanTicket(ticket1))
            self.assertFalse(self.__banManager.addBanTicket(ticket2))
            self.assertEqual(self.__banManager.size(), 1)
            banticket = self.__banManager.getTicketByID(ticket2.getID())
            self.assertEqual(banticket.getEndOfBanTime(defBanTime), ticket2.getEndOfBanTime(defBanTime))
            self.assertTrue(banticket.getEndOfBanTime(defBanTime) > prevEndOfBanTime)
            prevEndOfBanTime = ticket1.getEndOfBanTime(defBanTime)
            self.assertEqual(banticket.getTime(), 1167605999.0)
            if btnew == -1:
                self.assertEqual(banticket.getBanTime(defBanTime), -1)

    def testInListOK(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        ticket = BanTicket('193.168.0.128', 1167605999.0)
        self.assertTrue(self.__banManager._inBanList(ticket))

    def testInListNOK(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        ticket = BanTicket('111.111.1.111', 1167605999.0)
        self.assertFalse(self.__banManager._inBanList(ticket))

    def testBanTimeIncr(self):
        if False:
            return 10
        ticket = BanTicket(self.__ticket.getID(), self.__ticket.getTime())
        c = 0
        for i in (1000, 2000, -1):
            self.__banManager.addBanTicket(self.__ticket)
            c += 1
            ticket.setBanTime(i)
            self.assertFalse(self.__banManager.addBanTicket(ticket))
            self.assertEqual(str(self.__banManager.getTicketByID(ticket.getID())), 'BanTicket: ip=%s time=%s bantime=%s bancount=%s #attempts=0 matches=[]' % (ticket.getID(), ticket.getTime(), i, c))
        self.__banManager.addBanTicket(self.__ticket)
        c += 1
        ticket.setBanTime(-1)
        self.assertFalse(self.__banManager.addBanTicket(ticket))
        ticket.setBanTime(1000)
        self.assertFalse(self.__banManager.addBanTicket(ticket))
        self.assertEqual(str(self.__banManager.getTicketByID(ticket.getID())), 'BanTicket: ip=%s time=%s bantime=%s bancount=%s #attempts=0 matches=[]' % (ticket.getID(), ticket.getTime(), -1, c))

    def testUnban(self):
        if False:
            for i in range(10):
                print('nop')
        btime = self.__banManager.getBanTime()
        stime = self.__ticket.getTime()
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        self.assertTrue(self.__banManager._inBanList(self.__ticket))
        self.assertEqual(self.__banManager.unBanList(stime), [])
        self.assertEqual(self.__banManager.unBanList(stime + btime + 1), [self.__ticket])
        self.assertEqual(self.__banManager.size(), 0)
        self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
        ticket = BanTicket(self.__ticket.getID(), stime + 600)
        self.assertFalse(self.__banManager.addBanTicket(ticket))
        self.assertEqual(len(self.__banManager.unBanList(stime + btime + 1)), 0)
        self.assertEqual(len(self.__banManager.unBanList(stime + btime + 600 + 1)), 1)
        for i in range(5):
            ticket = BanTicket('193.168.0.%s' % i, stime)
            ticket.setBanTime(ticket.getBanTime(btime) + i * 10)
            self.assertTrue(self.__banManager.addBanTicket(ticket))
        self.assertEqual(len(self.__banManager.unBanList(stime + btime + 1 * 10 + 1)), 2)
        self.assertEqual(len(self.__banManager.unBanList(stime + btime + 5 * 10 + 1)), 3)
        self.assertEqual(self.__banManager.size(), 0)

    def testUnbanPermanent(self):
        if False:
            while True:
                i = 10
        btime = self.__banManager.getBanTime()
        self.__banManager.setBanTime(-1)
        try:
            self.assertTrue(self.__banManager.addBanTicket(self.__ticket))
            self.assertTrue(self.__banManager._inBanList(self.__ticket))
            self.assertEqual(self.__banManager.unBanList(self.__ticket.getTime() + btime + 1), [])
            self.assertEqual(self.__banManager.size(), 1)
        finally:
            self.__banManager.setBanTime(btime)

    def testBanList(self):
        if False:
            while True:
                i = 10
        tickets = [BanTicket('192.0.2.1', 1167605999.0), BanTicket('192.0.2.2', 1167605999.0)]
        tickets[1].setBanTime(-1)
        for t in tickets:
            self.__banManager.addBanTicket(t)
        self.assertSortedEqual(self.__banManager.getBanList(ordered=True, withTime=True), ['192.0.2.1 \t2006-12-31 23:59:59 + 600 = 2007-01-01 00:09:59', '192.0.2.2 \t2006-12-31 23:59:59 + -1 = 9999-12-31 23:59:59'])

class StatusExtendedCymruInfo(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Call before every test case.'
        super(StatusExtendedCymruInfo, self).setUp()
        unittest.F2B.SkipIfNoNetwork()
        setUpMyTime()
        self.__ban_ip = next(iter(DNSUtils.dnsToIp('resolver1.opendns.com')))
        self.__asn = '36692'
        self.__country = 'US'
        self.__rir = 'arin'
        ticket = BanTicket(self.__ban_ip, 1167605999.0)
        self.__banManager = BanManager()
        self.assertTrue(self.__banManager.addBanTicket(ticket))

    def tearDown(self):
        if False:
            return 10
        'Call after every test case.'
        super(StatusExtendedCymruInfo, self).tearDown()
        tearDownMyTime()
    available = (True, None)

    def _getBanListExtendedCymruInfo(self):
        if False:
            return 10
        tc = StatusExtendedCymruInfo
        if tc.available[0]:
            cymru_info = self.__banManager.getBanListExtendedCymruInfo(timeout=2 if unittest.F2B.fast else 20)
        else:
            cymru_info = tc.available[1]
        if cymru_info.get('error'):
            tc.available = (False, cymru_info)
            raise unittest.SkipTest('Skip test because service is not available: %s' % cymru_info['error'])
        return cymru_info

    def testCymruInfo(self):
        if False:
            i = 10
            return i + 15
        cymru_info = self._getBanListExtendedCymruInfo()
        self.assertDictEqual(cymru_info, {'asn': [self.__asn], 'country': [self.__country], 'rir': [self.__rir]})

    def testCymruInfoASN(self):
        if False:
            return 10
        self.assertEqual(self.__banManager.geBanListExtendedASN(self._getBanListExtendedCymruInfo()), [self.__asn])

    def testCymruInfoCountry(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.__banManager.geBanListExtendedCountry(self._getBanListExtendedCymruInfo()), [self.__country])

    def testCymruInfoRIR(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.__banManager.geBanListExtendedRIR(self._getBanListExtendedCymruInfo()), [self.__rir])

    def testCymruInfoNxdomain(self):
        if False:
            print('Hello World!')
        self.__banManager = BanManager()
        ticket = BanTicket('0.0.0.0', 1167605999.0)
        self.assertTrue(self.__banManager.addBanTicket(ticket))
        cymru_info = self._getBanListExtendedCymruInfo()
        self.assertDictEqual(cymru_info, {'asn': ['nxdomain'], 'country': ['nxdomain'], 'rir': ['nxdomain']})
        ticket = BanTicket('8.0.0.0', 1167606000.0)
        self.assertTrue(self.__banManager.addBanTicket(ticket))
        cymru_info = self._getBanListExtendedCymruInfo()
        self.assertSortedEqual(cymru_info, {'asn': ['nxdomain', '3356'], 'country': ['nxdomain', 'US'], 'rir': ['nxdomain', 'arin']}, level=-1, key=str)
__author__ = 'Serg G. Brester (sebres)'
__copyright__ = 'Copyright (c) 2015 Serg G. Brester, 2015- Fail2Ban Contributors'
__license__ = 'GPL'
from ..server.mytime import MyTime
import unittest
from ..server.ticket import Ticket, FailTicket, BanTicket

class TicketTests(unittest.TestCase):

    def testTicket(self):
        if False:
            for i in range(10):
                print('nop')
        tm = MyTime.time()
        matches = ['first', 'second']
        matches2 = ['first', 'second']
        matches3 = ['first', 'second', 'third']
        t = Ticket('193.168.0.128', tm, matches)
        self.assertEqual(t.getID(), '193.168.0.128')
        self.assertEqual(t.getIP(), '193.168.0.128')
        self.assertEqual(t.getTime(), tm)
        self.assertEqual(t.getMatches(), matches2)
        t.setAttempt(2)
        self.assertEqual(t.getAttempt(), 2)
        t.setBanCount(10)
        self.assertEqual(t.getBanCount(), 10)
        self.assertEqual(t.getBanTime(60 * 60), 60 * 60)
        self.assertFalse(t.isTimedOut(tm + 60 + 1, 60 * 60))
        self.assertTrue(t.isTimedOut(tm + 60 * 60 + 1, 60 * 60))
        t.setBanTime(60)
        self.assertEqual(t.getBanTime(60 * 60), 60)
        self.assertEqual(t.getBanTime(), 60)
        self.assertFalse(t.isTimedOut(tm))
        self.assertTrue(t.isTimedOut(tm + 60 + 1))
        t.setBanTime(-1)
        self.assertFalse(t.isTimedOut(tm + 60 + 1))
        t.setBanTime(60)
        tm = MyTime.time()
        matches = ['first', 'second']
        ft = FailTicket('193.168.0.128', tm, matches)
        ft.setBanTime(60 * 60)
        self.assertEqual(ft.getID(), '193.168.0.128')
        self.assertEqual(ft.getIP(), '193.168.0.128')
        self.assertEqual(ft.getTime(), tm)
        self.assertEqual(ft.getMatches(), matches2)
        ft.setAttempt(2)
        ft.setRetry(1)
        self.assertEqual(ft.getAttempt(), 2)
        self.assertEqual(ft.getRetry(), 1)
        ft.setRetry(2)
        self.assertEqual(ft.getRetry(), 2)
        ft.setRetry(3)
        self.assertEqual(ft.getRetry(), 3)
        ft.inc()
        self.assertEqual(ft.getAttempt(), 3)
        self.assertEqual(ft.getRetry(), 4)
        self.assertEqual(ft.getMatches(), matches2)
        ft.inc(['third'], 1, 10)
        self.assertEqual(ft.getAttempt(), 4)
        self.assertEqual(ft.getRetry(), 14)
        self.assertEqual(ft.getMatches(), matches3)
        self.assertEqual(ft.getTime(), tm)
        ft.adjustTime(tm - 60, 3600)
        self.assertEqual(ft.getTime(), tm)
        self.assertEqual(ft.getRetry(), 14)
        ft.adjustTime(tm + 60, 3600)
        self.assertEqual(ft.getTime(), tm + 60)
        self.assertEqual(ft.getRetry(), 14)
        ft.adjustTime(tm + 3600, 3600)
        self.assertEqual(ft.getTime(), tm + 3600)
        self.assertEqual(ft.getRetry(), 14)
        ft.adjustTime(tm + 7200, 3600)
        self.assertEqual(ft.getTime(), tm + 7200)
        self.assertEqual(ft.getRetry(), 7)
        self.assertEqual(ft.getAttempt(), 4)
        ft.setData('country', 'DE')
        self.assertEqual(ft.getData(), {'matches': ['first', 'second', 'third'], 'failures': 4, 'country': 'DE'})
        ft2 = FailTicket(ticket=ft)
        self.assertEqual(ft, ft2)
        self.assertEqual(ft.getData(), ft2.getData())
        self.assertEqual(ft2.getAttempt(), 4)
        self.assertEqual(ft2.getRetry(), 7)
        self.assertEqual(ft2.getMatches(), matches3)
        self.assertEqual(ft2.getTime(), ft.getTime())
        self.assertEqual(ft2.getTime(), ft.getTime())
        self.assertEqual(ft2.getBanTime(), ft.getBanTime())

    def testDiffIDAndIPTicket(self):
        if False:
            i = 10
            return i + 15
        tm = MyTime.time()
        t = Ticket('123-456-678', tm, data={'ip': '192.0.2.1'})
        self.assertEqual(t.getID(), '123-456-678')
        self.assertEqual(t.getIP(), '192.0.2.1')
        t = Ticket(('192.0.2.1', '5000'), tm, data={'ip': '192.0.2.1'})
        self.assertEqual(t.getID(), ('192.0.2.1', '5000'))
        self.assertEqual(t.getIP(), '192.0.2.1')

    def testTicketFlags(self):
        if False:
            return 10
        flags = ('restored', 'banned')
        ticket = Ticket('test', 0)
        trueflags = []
        for v in (True, False, True):
            for f in flags:
                setattr(ticket, f, v)
                if v:
                    trueflags.append(f)
                else:
                    trueflags.remove(f)
                for f2 in flags:
                    self.assertEqual(bool(getattr(ticket, f2)), f2 in trueflags)
        ticket = FailTicket(ticket=ticket)
        for f2 in flags:
            self.assertTrue(bool(getattr(ticket, f2)))

    def testTicketData(self):
        if False:
            i = 10
            return i + 15
        t = BanTicket('193.168.0.128', None, ['first', 'second'])
        t.setData('region', 'Hamburg', 'country', 'DE', 'city', 'Hamburg')
        self.assertEqual(t.getData(), {'matches': ['first', 'second'], 'failures': 0, 'region': 'Hamburg', 'country': 'DE', 'city': 'Hamburg'})
        t.setData({'region': None, 'country': 'FR', 'city': 'Paris'})
        self.assertEqual(t.getData(), {'city': 'Paris', 'country': 'FR'})
        t.setData({'region': 'Hamburg', 'country': 'DE', 'city': None})
        self.assertEqual(t.getData(), {'region': 'Hamburg', 'country': 'DE'})
        self.assertEqual(t.getData('region'), 'Hamburg')
        self.assertEqual(t.getData('country'), 'DE')
        t.setData(region='Bremen', city='Bremen')
        self.assertEqual(t.getData(), {'region': 'Bremen', 'country': 'DE', 'city': 'Bremen'})
        t.setData('region', 'Brandenburg', 'city', 'Berlin')
        self.assertEqual(t.getData('region'), 'Brandenburg')
        self.assertEqual(t.getData('city'), 'Berlin')
        self.assertEqual(t.getData(), {'city': 'Berlin', 'region': 'Brandenburg', 'country': 'DE'})
        self.assertEqual(t.getData(('city', 'country')), {'city': 'Berlin', 'country': 'DE'})
        self.assertEqual(t.getData(lambda k: k.upper() == 'COUNTRY'), {'country': 'DE'})
        t.setData('city', None)
        self.assertEqual(t.getData(), {'region': 'Brandenburg', 'country': 'DE'})
        self.assertEqual(t.getData('city', 'Unknown'), 'Unknown')
        t.setData('continent', 'Europe')
        t.setData(*['country', 'RU', 'region', 'Moscow'])
        self.assertEqual(t.getData(), {'continent': 'Europe', 'country': 'RU', 'region': 'Moscow'})
        t.setData({})
        self.assertEqual(t.getData(), {})
        self.assertEqual(t.getData('anything', 'default'), 'default')
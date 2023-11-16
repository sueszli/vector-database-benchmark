__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
from threading import Lock
from .ticket import BanTicket
from .mytime import MyTime
from ..helpers import getLogger, logging
logSys = getLogger(__name__)

class BanManager:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__lock = Lock()
        self.__banList = dict()
        self.__banTime = 600
        self.__banTotal = 0
        self._nextUnbanTime = BanTicket.MAX_TIME

    def setBanTime(self, value):
        if False:
            return 10
        self.__banTime = int(value)

    def getBanTime(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__banTime

    def setBanTotal(self, value):
        if False:
            i = 10
            return i + 15
        self.__banTotal = value

    def getBanTotal(self):
        if False:
            while True:
                i = 10
        return self.__banTotal

    def getBanList(self, ordered=False, withTime=False):
        if False:
            i = 10
            return i + 15
        if not ordered:
            return list(self.__banList.keys())
        with self.__lock:
            lst = []
            for ticket in self.__banList.values():
                eob = ticket.getEndOfBanTime(self.__banTime)
                lst.append((ticket, eob))
        lst.sort(key=lambda t: t[1])
        t2s = MyTime.time2str
        if withTime:
            return ['%s \t%s + %d = %s' % (t[0].getID(), t2s(t[0].getTime()), t[0].getBanTime(self.__banTime), t2s(t[1])) for t in lst]
        return [t[0].getID() for t in lst]

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(list(self.__banList.values()))

    @staticmethod
    def handleBlankResult(value):
        if False:
            print('Hello World!')
        if value is None or len(value) == 0:
            return 'unknown'
        else:
            return value

    def getBanListExtendedCymruInfo(self, timeout=10):
        if False:
            return 10
        return_dict = {'asn': [], 'country': [], 'rir': []}
        if not hasattr(self, 'dnsResolver'):
            global dns
            try:
                import dns.exception
                import dns.resolver
                resolver = dns.resolver.Resolver()
                resolver.lifetime = timeout
                resolver.timeout = timeout / 2
                self.dnsResolver = resolver
            except ImportError as e:
                logSys.error('dnspython package is required but could not be imported')
                return_dict['error'] = repr(e)
                return_dict['asn'].append('error')
                return_dict['country'].append('error')
                return_dict['rir'].append('error')
                return return_dict
        with self.__lock:
            banIPs = [banData.getIP() for banData in list(self.__banList.values())]
        try:
            for ip in banIPs:
                question = ip.getPTR('origin.asn.cymru.com' if ip.isIPv4 else 'origin6.asn.cymru.com')
                try:
                    resolver = self.dnsResolver
                    answers = resolver.query(question, 'TXT')
                    if not answers:
                        raise ValueError('No data retrieved')
                    asns = set()
                    countries = set()
                    rirs = set()
                    for rdata in answers:
                        (asn, net, country, rir, changed) = [answer.strip('\'" ') for answer in rdata.to_text().split('|')]
                        asn = self.handleBlankResult(asn)
                        country = self.handleBlankResult(country)
                        rir = self.handleBlankResult(rir)
                        asns.add(self.handleBlankResult(asn))
                        countries.add(self.handleBlankResult(country))
                        rirs.add(self.handleBlankResult(rir))
                    return_dict['asn'].append(', '.join(sorted(asns)))
                    return_dict['country'].append(', '.join(sorted(countries)))
                    return_dict['rir'].append(', '.join(sorted(rirs)))
                except dns.resolver.NXDOMAIN:
                    return_dict['asn'].append('nxdomain')
                    return_dict['country'].append('nxdomain')
                    return_dict['rir'].append('nxdomain')
                except (dns.exception.DNSException, dns.resolver.NoNameservers, dns.exception.Timeout) as dnse:
                    logSys.error('DNSException %r querying Cymru for %s TXT', dnse, question)
                    if logSys.level <= logging.DEBUG:
                        logSys.exception(dnse)
                    return_dict['error'] = repr(dnse)
                    break
                except Exception as e:
                    logSys.error('Unhandled Exception %r querying Cymru for %s TXT', e, question)
                    if logSys.level <= logging.DEBUG:
                        logSys.exception(e)
                    return_dict['error'] = repr(e)
                    break
        except Exception as e:
            logSys.error('Failure looking up extended Cymru info: %s', e)
            if logSys.level <= logging.DEBUG:
                logSys.exception(e)
            return_dict['error'] = repr(e)
        return return_dict

    def geBanListExtendedASN(self, cymru_info):
        if False:
            for i in range(10):
                print('nop')
        try:
            return [asn for asn in cymru_info['asn']]
        except Exception as e:
            logSys.error('Failed to lookup ASN')
            logSys.exception(e)
            return []

    def geBanListExtendedCountry(self, cymru_info):
        if False:
            for i in range(10):
                print('nop')
        try:
            return [country for country in cymru_info['country']]
        except Exception as e:
            logSys.error('Failed to lookup Country')
            logSys.exception(e)
            return []

    def geBanListExtendedRIR(self, cymru_info):
        if False:
            while True:
                i = 10
        try:
            return [rir for rir in cymru_info['rir']]
        except Exception as e:
            logSys.error('Failed to lookup RIR')
            logSys.exception(e)
            return []

    def addBanTicket(self, ticket, reason={}):
        if False:
            print('Hello World!')
        eob = ticket.getEndOfBanTime(self.__banTime)
        if eob < MyTime.time():
            reason['expired'] = 1
            return False
        with self.__lock:
            fid = ticket.getID()
            oldticket = self.__banList.get(fid)
            if oldticket:
                reason['ticket'] = oldticket
                if eob > oldticket.getEndOfBanTime(self.__banTime):
                    reason['prolong'] = 1
                    btm = ticket.getBanTime(self.__banTime)
                    if btm != -1:
                        diftm = ticket.getTime() - oldticket.getTime()
                        if diftm > 0:
                            btm += diftm
                    oldticket.setBanTime(btm)
                return False
            self.__banList[fid] = ticket
            self.__banTotal += 1
            ticket.incrBanCount()
            if self._nextUnbanTime > eob:
                self._nextUnbanTime = eob
            return True

    def size(self):
        if False:
            return 10
        return len(self.__banList)

    def _inBanList(self, ticket):
        if False:
            print('Hello World!')
        return ticket.getID() in self.__banList

    def unBanList(self, time, maxCount=2147483647):
        if False:
            for i in range(10):
                print('nop')
        with self.__lock:
            nextUnbanTime = self._nextUnbanTime
            if nextUnbanTime > time:
                return list()
            unBanList = {}
            nextUnbanTime = BanTicket.MAX_TIME
            for (fid, ticket) in self.__banList.items():
                eob = ticket.getEndOfBanTime(self.__banTime)
                if time > eob:
                    unBanList[fid] = ticket
                    if len(unBanList) >= maxCount:
                        nextUnbanTime = self._nextUnbanTime
                        break
                elif nextUnbanTime > eob:
                    nextUnbanTime = eob
            self._nextUnbanTime = nextUnbanTime
            if len(unBanList):
                if len(unBanList) / 2.0 <= len(self.__banList) / 3.0:
                    for fid in unBanList.keys():
                        del self.__banList[fid]
                else:
                    self.__banList = dict(((fid, ticket) for (fid, ticket) in self.__banList.items() if fid not in unBanList))
            return list(unBanList.values())

    def flushBanList(self):
        if False:
            print('Hello World!')
        with self.__lock:
            uBList = list(self.__banList.values())
            self.__banList = dict()
            return uBList

    def getTicketByID(self, fid):
        if False:
            i = 10
            return i + 15
        with self.__lock:
            try:
                return self.__banList.pop(fid)
            except KeyError:
                pass
        return None
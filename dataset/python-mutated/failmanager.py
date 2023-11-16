__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
from threading import Lock
import logging
from .ticket import FailTicket, BanTicket
from ..helpers import getLogger, BgService
logSys = getLogger(__name__)
logLevel = logging.DEBUG

class FailManager:

    def __init__(self):
        if False:
            return 10
        self.__lock = Lock()
        self.__failList = dict()
        self.__maxRetry = 3
        self.__maxTime = 600
        self.__failTotal = 0
        self.maxMatches = 5
        self.__bgSvc = BgService()

    def setFailTotal(self, value):
        if False:
            i = 10
            return i + 15
        self.__failTotal = value

    def getFailTotal(self):
        if False:
            while True:
                i = 10
        return self.__failTotal

    def getFailCount(self):
        if False:
            for i in range(10):
                print('nop')
        with self.__lock:
            return (len(self.__failList), sum([f.getRetry() for f in list(self.__failList.values())]))

    def setMaxRetry(self, value):
        if False:
            return 10
        self.__maxRetry = value

    def getMaxRetry(self):
        if False:
            i = 10
            return i + 15
        return self.__maxRetry

    def setMaxTime(self, value):
        if False:
            i = 10
            return i + 15
        self.__maxTime = value

    def getMaxTime(self):
        if False:
            return 10
        return self.__maxTime

    def addFailure(self, ticket, count=1, observed=False):
        if False:
            for i in range(10):
                print('nop')
        attempts = 1
        with self.__lock:
            fid = ticket.getID()
            try:
                fData = self.__failList[fid]
                if fData is ticket:
                    matches = None
                    attempt = 1
                else:
                    matches = ticket.getMatches() if self.maxMatches else None
                    attempt = ticket.getAttempt()
                    if attempt <= 0:
                        attempt += 1
                unixTime = ticket.getTime()
                fData.adjustTime(unixTime, self.__maxTime)
                fData.inc(matches, attempt, count)
                if self.maxMatches:
                    matches = fData.getMatches()
                    if len(matches) > self.maxMatches:
                        fData.setMatches(matches[-self.maxMatches:])
                else:
                    fData.setMatches(None)
            except KeyError:
                if observed or isinstance(ticket, BanTicket):
                    return ticket.getRetry()
                if isinstance(ticket, FailTicket):
                    fData = ticket
                else:
                    fData = FailTicket.wrap(ticket)
                if count > ticket.getAttempt():
                    fData.setRetry(count)
                self.__failList[fid] = fData
            attempts = fData.getRetry()
            self.__failTotal += 1
            if logSys.getEffectiveLevel() <= logLevel:
                failures_summary = ', '.join(['%s:%d' % (k, v.getRetry()) for (k, v) in self.__failList.items()])
                logSys.log(logLevel, 'Total # of detected failures: %d. Current failures from %d IPs (IP:count): %s' % (self.__failTotal, len(self.__failList), failures_summary))
        self.__bgSvc.service()
        return attempts

    def size(self):
        if False:
            return 10
        return len(self.__failList)

    def cleanup(self, time):
        if False:
            while True:
                i = 10
        time -= self.__maxTime
        with self.__lock:
            todelete = [fid for (fid, item) in self.__failList.items() if item.getTime() <= time]
            if len(todelete) == len(self.__failList):
                self.__failList = dict()
            elif not len(todelete):
                return
            if len(todelete) / 2.0 <= len(self.__failList) / 3.0:
                for fid in todelete:
                    del self.__failList[fid]
            else:
                self.__failList = dict(((fid, item) for (fid, item) in self.__failList.items() if item.getTime() > time))
        self.__bgSvc.service()

    def delFailure(self, fid):
        if False:
            for i in range(10):
                print('nop')
        with self.__lock:
            try:
                del self.__failList[fid]
            except KeyError:
                pass

    def toBan(self, fid=None):
        if False:
            while True:
                i = 10
        with self.__lock:
            for fid in [fid] if fid is not None and fid in self.__failList else self.__failList:
                data = self.__failList[fid]
                if data.getRetry() >= self.__maxRetry:
                    del self.__failList[fid]
                    return data
        self.__bgSvc.service()
        raise FailManagerEmpty

class FailManagerEmpty(Exception):
    pass
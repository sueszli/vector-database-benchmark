__author__ = 'Steven Hiscocks'
__copyright__ = 'Copyright (c) 2013 Steven Hiscocks'
__license__ = 'GPL'
import os
import time
from distutils.version import LooseVersion
from systemd import journal
if LooseVersion(getattr(journal, '__version__', '0')) < '204':
    raise ImportError('Fail2Ban requires systemd >= 204')
from .failmanager import FailManagerEmpty
from .filter import JournalFilter, Filter
from .mytime import MyTime
from .utils import Utils
from ..helpers import getLogger, logging, splitwords, uni_decode
logSys = getLogger(__name__)

class FilterSystemd(JournalFilter):

    def __init__(self, jail, **kwargs):
        if False:
            print('Hello World!')
        jrnlargs = FilterSystemd._getJournalArgs(kwargs)
        JournalFilter.__init__(self, jail, **kwargs)
        self.__modified = 0
        self.__journal = journal.Reader(**jrnlargs)
        self.__matches = []
        self.setDatePattern(None)
        logSys.debug('Created FilterSystemd')

    @staticmethod
    def _getJournalArgs(kwargs):
        if False:
            i = 10
            return i + 15
        args = {'converters': {'__CURSOR': lambda x: x}}
        try:
            args['path'] = kwargs.pop('journalpath')
        except KeyError:
            pass
        try:
            args['files'] = kwargs.pop('journalfiles')
        except KeyError:
            pass
        else:
            import glob
            p = args['files']
            if not isinstance(p, (list, set, tuple)):
                p = splitwords(p)
            files = []
            for p in p:
                files.extend(glob.glob(p))
            args['files'] = list(set(files))
        try:
            args['flags'] = int(kwargs.pop('journalflags'))
        except KeyError:
            if ('files' not in args or not len(args['files'])) and ('path' not in args or not args['path']):
                args['flags'] = int(os.getenv('F2B_SYSTEMD_DEFAULT_FLAGS', 4))
        try:
            args['namespace'] = kwargs.pop('namespace')
        except KeyError:
            pass
        return args

    def _addJournalMatches(self, matches):
        if False:
            print('Hello World!')
        if self.__matches:
            self.__journal.add_disjunction()
        newMatches = []
        for match in matches:
            newMatches.append([])
            for match_element in match:
                self.__journal.add_match(match_element)
                newMatches[-1].append(match_element)
            self.__journal.add_disjunction()
        self.__matches.extend(newMatches)

    def addJournalMatch(self, match):
        if False:
            while True:
                i = 10
        newMatches = [[]]
        for match_element in match:
            if match_element == '+':
                newMatches.append([])
            else:
                newMatches[-1].append(match_element)
        try:
            self._addJournalMatches(newMatches)
        except ValueError:
            logSys.error('Error adding journal match for: %r', ' '.join(match))
            self.resetJournalMatches()
            raise
        else:
            logSys.info('[%s] Added journal match for: %r', self.jailName, ' '.join(match))

    def resetJournalMatches(self):
        if False:
            while True:
                i = 10
        self.__journal.flush_matches()
        logSys.debug('[%s] Flushed all journal matches', self.jailName)
        match_copy = self.__matches[:]
        self.__matches = []
        try:
            self._addJournalMatches(match_copy)
        except ValueError:
            logSys.error('Error restoring journal matches')
            raise
        else:
            logSys.debug('Journal matches restored')

    def delJournalMatch(self, match=None):
        if False:
            i = 10
            return i + 15
        if match is None:
            if not self.__matches:
                return
            del self.__matches[:]
        elif match in self.__matches:
            del self.__matches[self.__matches.index(match)]
        else:
            raise ValueError('Match %r not found' % match)
        self.resetJournalMatches()
        logSys.info('[%s] Removed journal match for: %r', self.jailName, match if match else '*')

    def getJournalMatch(self):
        if False:
            return 10
        return self.__matches

    def getJournalReader(self):
        if False:
            i = 10
            return i + 15
        return self.__journal

    def getJrnEntTime(self, logentry):
        if False:
            for i in range(10):
                print('nop')
        ' Returns time of entry as tuple (ISO-str, Posix).'
        date = logentry.get('_SOURCE_REALTIME_TIMESTAMP')
        if date is None:
            date = logentry.get('__REALTIME_TIMESTAMP')
        return (date.isoformat(), time.mktime(date.timetuple()) + date.microsecond / 1000000.0)

    def formatJournalEntry(self, logentry):
        if False:
            for i in range(10):
                print('nop')
        enc = self.getLogEncoding()
        logelements = []
        v = logentry.get('_HOSTNAME')
        if v:
            logelements.append(uni_decode(v, enc))
        v = logentry.get('SYSLOG_IDENTIFIER')
        if not v:
            v = logentry.get('_COMM')
        if v:
            logelements.append(uni_decode(v, enc))
            v = logentry.get('SYSLOG_PID')
            if not v:
                v = logentry.get('_PID')
            if v:
                try:
                    v = '[%i]' % v
                except TypeError:
                    try:
                        v = '[%i]' % int(v, 0)
                    except (TypeError, ValueError):
                        v = '[%s]' % v
                logelements[-1] += v
            logelements[-1] += ':'
            if logelements[-1] == 'kernel:':
                monotonic = logentry.get('_SOURCE_MONOTONIC_TIMESTAMP')
                if monotonic is None:
                    monotonic = logentry.get('__MONOTONIC_TIMESTAMP')[0]
                logelements.append('[%12.6f]' % monotonic.total_seconds())
        msg = logentry.get('MESSAGE', '')
        if isinstance(msg, list):
            logelements.append(' '.join((uni_decode(v, enc) for v in msg)))
        else:
            logelements.append(uni_decode(msg, enc))
        logline = ' '.join(logelements)
        date = self.getJrnEntTime(logentry)
        logSys.log(5, '[%s] Read systemd journal entry: %s %s', self.jailName, date[0], logline)
        return ((logline[:0], date[0] + ' ', logline.replace('\n', '\\n')), date[1])

    def seekToTime(self, date):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(date, int):
            date = float(date)
        self.__journal.seek_realtime(date)

    def inOperationMode(self):
        if False:
            return 10
        self.inOperation = True
        logSys.info('[%s] Jail is in operation now (process new journal entries)', self.jailName)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.getJournalMatch():
            logSys.notice("[%s] Jail started without 'journalmatch' set. Jail regexs will be checked against all journal entries, which is not advised for performance reasons.", self.jailName)
        logentry = None
        try:
            self.__journal.seek_tail()
            logentry = self.__journal.get_previous()
            if logentry:
                self.__journal.get_next()
        except OSError:
            logentry = None
        if logentry:
            startTime = 0
            if self.jail.database is not None:
                startTime = self.jail.database.getJournalPos(self.jail, 'systemd-journal') or 0
            startTime = max(startTime, MyTime.time() - int(self.getFindTime()))
            self.seekToTime(startTime)
            self.inOperation = False
            startTime = (1, MyTime.time(), logentry.get('__CURSOR'))
        else:
            self.inOperationMode()
            startTime = MyTime.time()
            self.seekToTime(startTime)
            startTime = (0, startTime)
        try:
            self.__journal.get_previous()
        except OSError:
            pass
        wcode = journal.NOP
        line = None
        while self.active:
            try:
                if wcode == journal.NOP and self.inOperation:
                    wcode = Utils.wait_for(lambda : not self.active and journal.APPEND or self.__journal.wait(Utils.DEFAULT_SLEEP_INTERVAL), self.sleeptime, 1e-05)
                    if self.active and wcode == journal.INVALIDATE:
                        if self.ticks:
                            logSys.log(logging.DEBUG, '[%s] Invalidate signaled, take a little break (rotation ends)', self.jailName)
                            time.sleep(self.sleeptime * 0.25)
                        Utils.wait_for(lambda : not self.active or self.__journal.wait(Utils.DEFAULT_SLEEP_INTERVAL) != journal.INVALIDATE, self.sleeptime * 3, 1e-05)
                        if self.ticks:
                            try:
                                if self.__journal.get_previous():
                                    self.__journal.get_next()
                            except OSError:
                                pass
                if self.idle:
                    if not Utils.wait_for(lambda : not self.active or not self.idle, self.sleeptime * 10, self.sleeptime):
                        self.ticks += 1
                        continue
                self.__modified = 0
                while self.active:
                    logentry = None
                    try:
                        logentry = self.__journal.get_next()
                    except OSError as e:
                        logSys.error('Error reading line from systemd journal: %s', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
                    self.ticks += 1
                    if logentry:
                        (line, tm) = self.formatJournalEntry(logentry)
                        if not self.inOperation:
                            if tm >= MyTime.time() - 1:
                                self.inOperationMode()
                            elif startTime[0] == 1:
                                if logentry.get('__CURSOR') == startTime[2] or tm > startTime[1]:
                                    startTime = (0, MyTime.time() * 2 - startTime[1])
                            elif tm > startTime[1]:
                                self.inOperationMode()
                        self.processLineAndAdd(line, tm)
                        self.__modified += 1
                        if self.__modified >= 100:
                            wcode = journal.APPEND
                            break
                    else:
                        if not self.inOperation:
                            self.inOperationMode()
                        wcode = journal.NOP
                        break
                self.__modified = 0
                if self.ticks % 10 == 0:
                    self.performSvc()
                if self.jail.database:
                    if line:
                        self._pendDBUpdates['systemd-journal'] = (tm, line[1])
                        line = None
                    if self._pendDBUpdates and (self.ticks % 100 == 0 or MyTime.time() >= self._nextUpdateTM or (not self.active)):
                        self._updateDBPending()
                        self._nextUpdateTM = MyTime.time() + Utils.DEFAULT_SLEEP_TIME * 5
            except Exception as e:
                if not self.active:
                    break
                wcode = journal.NOP
                logSys.error('Caught unhandled exception in main cycle: %r', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
                self.commonError('unhandled', e)
        logSys.debug('[%s] filter terminated', self.jailName)
        self.closeJournal()
        logSys.debug('[%s] filter exited (systemd)', self.jailName)
        return True

    def closeJournal(self):
        if False:
            print('Hello World!')
        try:
            (jnl, self.__journal) = (self.__journal, None)
            if jnl:
                jnl.close()
        except Exception as e:
            logSys.error('Close journal failed: %r', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)

    def status(self, flavor='basic'):
        if False:
            while True:
                i = 10
        ret = super(FilterSystemd, self).status(flavor=flavor)
        ret.append(('Journal matches', [' + '.join((' '.join(match) for match in self.__matches))]))
        return ret

    def _updateDBPending(self):
        if False:
            i = 10
            return i + 15
        'Apply pending updates (jornal position) to database.\n\t\t'
        db = self.jail.database
        while True:
            try:
                (log, args) = self._pendDBUpdates.popitem()
            except KeyError:
                break
            db.updateJournal(self.jail, log, *args)

    def onStop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop monitoring of journal. Invoked after run method.\n\t\t'
        self.closeJournal()
        if self._pendDBUpdates and self.jail.database:
            self._updateDBPending()
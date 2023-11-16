__author__ = 'Cyril Jaquier, Lee Clemens, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier, 2011-2012 Lee Clemens, 2012 Yaroslav Halchenko'
__license__ = 'GPL'
import logging
import math
import random
import queue
from .actions import Actions
from ..helpers import getLogger, _as_bool, extractOptions, MyTime
from .mytime import MyTime
logSys = getLogger(__name__)

class Jail(object):
    """Fail2Ban jail, which manages a filter and associated actions.

	The class handles the initialisation of a filter, and actions. It's
	role is then to act as an interface between the filter and actions,
	passing bans detected by the filter, for the actions to then act upon.

	Parameters
	----------
	name : str
		Name assigned to the jail.
	backend : str
		Backend to be used for filter. "auto" will attempt to pick
		the most preferred backend method. Default: "auto"
	db : Fail2BanDb
		Fail2Ban persistent database instance. Default: `None`

	Attributes
	----------
	name
	database
	filter
	actions
	idle
	status
	"""
    _BACKENDS = ['pyinotify', 'polling', 'systemd']

    def __init__(self, name, backend='auto', db=None):
        if False:
            print('Hello World!')
        self.__db = db
        if len(name) >= 26:
            logSys.warning('Jail name %r might be too long and some commands might not function correctly. Please shorten' % name)
        self.__name = name
        self.__queue = queue.Queue()
        self.__filter = None
        self._banExtra = {}
        logSys.info("Creating new jail '%s'" % self.name)
        if backend is not None:
            self._setBackend(backend)
        self.backend = backend

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r)' % (self.__class__.__name__, self.name)

    def _setBackend(self, backend):
        if False:
            print('Hello World!')
        (backend, beArgs) = extractOptions(backend)
        backend = backend.lower()
        backends = self._BACKENDS
        if backend != 'auto':
            if not backend in self._BACKENDS:
                logSys.error("Unknown backend %s. Must be among %s or 'auto'" % (backend, backends))
                raise ValueError("Unknown backend %s. Must be among %s or 'auto'" % (backend, backends))
            backends = backends[backends.index(backend):]
        for b in backends:
            initmethod = getattr(self, '_init%s' % b.capitalize())
            try:
                initmethod(**beArgs)
                if backend != 'auto' and b != backend:
                    logSys.warning('Could only initiated %r backend whenever %r was requested' % (b, backend))
                else:
                    logSys.info('Initiated %r backend' % b)
                self.__actions = Actions(self)
                return
            except ImportError as e:
                logSys.log(logging.DEBUG if backend == 'auto' else logging.ERROR, 'Backend %r failed to initialize due to %s' % (b, e))
        logSys.error('Failed to initialize any backend for Jail %r' % self.name)
        raise RuntimeError('Failed to initialize any backend for Jail %r' % self.name)

    def _initPolling(self, **kwargs):
        if False:
            while True:
                i = 10
        from .filterpoll import FilterPoll
        logSys.info("Jail '%s' uses poller %r" % (self.name, kwargs))
        self.__filter = FilterPoll(self, **kwargs)

    def _initPyinotify(self, **kwargs):
        if False:
            return 10
        from .filterpyinotify import FilterPyinotify
        logSys.info("Jail '%s' uses pyinotify %r" % (self.name, kwargs))
        self.__filter = FilterPyinotify(self, **kwargs)

    def _initSystemd(self, **kwargs):
        if False:
            print('Hello World!')
        from .filtersystemd import FilterSystemd
        logSys.info("Jail '%s' uses systemd %r" % (self.name, kwargs))
        self.__filter = FilterSystemd(self, **kwargs)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'Name of jail.\n\t\t'
        return self.__name

    @property
    def database(self):
        if False:
            while True:
                i = 10
        'The database used to store persistent data for the jail.\n\t\t'
        return self.__db

    @database.setter
    def database(self, value):
        if False:
            return 10
        self.__db = value

    @property
    def filter(self):
        if False:
            for i in range(10):
                print('nop')
        'The filter which the jail is using to monitor log files.\n\t\t'
        return self.__filter

    @property
    def actions(self):
        if False:
            return 10
        'Actions object used to manage actions for jail.\n\t\t'
        return self.__actions

    @property
    def idle(self):
        if False:
            i = 10
            return i + 15
        'A boolean indicating whether jail is idle.\n\t\t'
        return self.filter.idle or self.actions.idle

    @idle.setter
    def idle(self, value):
        if False:
            i = 10
            return i + 15
        self.filter.idle = value
        self.actions.idle = value

    def status(self, flavor='basic'):
        if False:
            i = 10
            return i + 15
        'The status of the jail.\n\t\t'
        return [('Filter', self.filter.status(flavor=flavor)), ('Actions', self.actions.status(flavor=flavor))]

    @property
    def hasFailTickets(self):
        if False:
            print('Hello World!')
        'Retrieve whether queue has tickets to ban.\n\t\t'
        return not self.__queue.empty()

    def putFailTicket(self, ticket):
        if False:
            print('Hello World!')
        'Add a fail ticket to the jail.\n\n\t\tUsed by filter to add a failure for banning.\n\t\t'
        self.__queue.put(ticket)

    def getFailTicket(self):
        if False:
            return 10
        'Get a fail ticket from the jail.\n\n\t\tUsed by actions to get a failure for banning.\n\t\t'
        try:
            ticket = self.__queue.get(False)
            return ticket
        except queue.Empty:
            return False

    def setBanTimeExtra(self, opt, value):
        if False:
            return 10
        be = self._banExtra
        if value == '':
            value = None
        if value is not None:
            be[opt] = value
        elif opt in be:
            del be[opt]
        logSys.info('Set banTime.%s = %s', opt, value)
        if opt == 'increment':
            be[opt] = _as_bool(value)
            if be.get(opt) and self.database is None:
                logSys.warning('ban time increment is not available as long jail database is not set')
        if opt in ['maxtime', 'rndtime']:
            if not value is None:
                be[opt] = MyTime.str2seconds(value)
        if opt in ['formula', 'factor', 'maxtime', 'rndtime', 'multipliers'] or be.get('evformula', None) is None:
            if opt == 'multipliers':
                be['evmultipliers'] = [int(i) for i in (value.split(' ') if value is not None and value != '' else [])]
            multipliers = be.get('evmultipliers', [])
            banFactor = eval(be.get('factor', '1'))
            if len(multipliers):
                evformula = lambda ban, banFactor=banFactor: ban.Time * banFactor * multipliers[ban.Count if ban.Count < len(multipliers) else -1]
            else:
                formula = be.get('formula', 'ban.Time * (1<<(ban.Count if ban.Count<20 else 20)) * banFactor')
                formula = compile(formula, '~inline-conf-expr~', 'eval')
                evformula = lambda ban, banFactor=banFactor, formula=formula: max(ban.Time, eval(formula))
            if not be.get('maxtime', None) is None:
                maxtime = be['maxtime']
                evformula = lambda ban, evformula=evformula: min(evformula(ban), maxtime)
            if not be.get('rndtime', None) is None:
                rndtime = be['rndtime']
                evformula = lambda ban, evformula=evformula: evformula(ban) + random.random() * rndtime
            be['evformula'] = evformula

    def getBanTimeExtra(self, opt=None):
        if False:
            i = 10
            return i + 15
        if opt is not None:
            return self._banExtra.get(opt, None)
        return self._banExtra

    def getMaxBanTime(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns max possible ban-time of jail.\n\t\t'
        return self._banExtra.get('maxtime', -1) if self._banExtra.get('increment') else self.actions.getBanTime()

    def restoreCurrentBans(self, correctBanTime=True):
        if False:
            return 10
        'Restore any previous valid bans from the database.\n\t\t'
        try:
            if self.database is not None:
                if self._banExtra.get('increment'):
                    forbantime = None
                    if correctBanTime:
                        correctBanTime = self.getMaxBanTime()
                else:
                    forbantime = self.actions.getBanTime()
                for ticket in self.database.getCurrentBans(jail=self, forbantime=forbantime, correctBanTime=correctBanTime, maxmatches=self.filter.failManager.maxMatches):
                    try:
                        if self.filter._inIgnoreIPList(ticket.getID(), ticket):
                            continue
                        ticket.restored = True
                        btm = ticket.getBanTime(forbantime)
                        diftm = MyTime.time() - ticket.getTime()
                        if btm != -1 and diftm > 0:
                            btm -= diftm
                        if btm != -1 and btm <= 0:
                            continue
                        self.putFailTicket(ticket)
                    except Exception as e:
                        logSys.error('Restore ticket failed: %s', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
        except Exception as e:
            logSys.error('Restore bans failed: %s', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start the jail, by starting filter and actions threads.\n\n\t\tOnce stated, also queries the persistent database to reinstate\n\t\tany valid bans.\n\t\t'
        logSys.debug('Starting jail %r', self.name)
        self.filter.start()
        self.actions.start()
        self.restoreCurrentBans()
        logSys.info('Jail %r started', self.name)

    def stop(self, stop=True, join=True):
        if False:
            print('Hello World!')
        'Stop the jail, by stopping filter and actions threads.\n\t\t'
        if stop:
            logSys.debug('Stopping jail %r', self.name)
        for obj in (self.filter, self.actions):
            try:
                if stop:
                    obj.stop()
                if join:
                    obj.join()
            except Exception as e:
                logSys.error('Stop %r of jail %r failed: %s', obj, self.name, e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
        if join:
            logSys.info('Jail %r stopped', self.name)

    def isAlive(self):
        if False:
            return 10
        'Check jail "isAlive" by checking filter and actions threads.\n\t\t'
        return self.filter.isAlive() or self.actions.isAlive()
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from copy import copy
from datetime import date, datetime, timedelta
import inspect
import itertools
import random
import threading
import time
from ib.ext.Contract import Contract
import ib.opt as ibopt
from backtrader import TimeFrame, Position
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import bytes, bstr, queue, with_metaclass, long
from backtrader.utils import AutoDict, UTC
bytes = bstr

def _ts2dt(tstamp=None):
    if False:
        return 10
    if not tstamp:
        return datetime.utcnow()
    (sec, msec) = divmod(long(tstamp), 1000)
    usec = msec * 1000
    return datetime.utcfromtimestamp(sec).replace(microsecond=usec)

class RTVolume(object):
    """Parses a tickString tickType 48 (RTVolume) event from the IB API into its
    constituent fields

    Supports using a "price" to simulate an RTVolume from a tickPrice event
    """
    _fields = [('price', float), ('size', int), ('datetime', _ts2dt), ('volume', int), ('vwap', float), ('single', bool)]

    def __init__(self, rtvol='', price=None, tmoffset=None):
        if False:
            while True:
                i = 10
        tokens = iter(rtvol.split(';'))
        for (name, func) in self._fields:
            setattr(self, name, func(next(tokens)) if rtvol else func())
        if price is not None:
            self.price = price
        if tmoffset is not None:
            self.datetime += tmoffset

class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._singleton

def ibregister(f):
    if False:
        i = 10
        return i + 15
    f._ibregister = True
    return f

class IBStore(with_metaclass(MetaSingleton, object)):
    """Singleton class wrapping an ibpy ibConnection instance.

    The parameters can also be specified in the classes which use this store,
    like ``IBData`` and ``IBBroker``

    Params:

      - ``host`` (default:``127.0.0.1``): where IB TWS or IB Gateway are
        actually running. And although this will usually be the localhost, it
        must not be

      - ``port`` (default: ``7496``): port to connect to. The demo system uses
        ``7497``

      - ``clientId`` (default: ``None``): which clientId to use to connect to
        TWS.

        ``None``: generates a random id between 1 and 65535
        An ``integer``: will be passed as the value to use.

      - ``notifyall`` (default: ``False``)

        If ``False`` only ``error`` messages will be sent to the
        ``notify_store`` methods of ``Cerebro`` and ``Strategy``.

        If ``True``, each and every message received from TWS will be notified

      - ``_debug`` (default: ``False``)

        Print all messages received from TWS to standard output

      - ``reconnect`` (default: ``3``)

        Number of attempts to try to reconnect after the 1st connection attempt
        fails

        Set it to a ``-1`` value to keep on reconnecting forever

      - ``timeout`` (default: ``3.0``)

        Time in seconds between reconnection attemps

      - ``timeoffset`` (default: ``True``)

        If True, the time obtained from ``reqCurrentTime`` (IB Server time)
        will be used to calculate the offset to localtime and this offset will
        be used for the price notifications (tickPrice events, for example for
        CASH markets) to modify the locally calculated timestamp.

        The time offset will propagate to other parts of the ``backtrader``
        ecosystem like the **resampling** to align resampling timestamps using
        the calculated offset.

      - ``timerefresh`` (default: ``60.0``)

        Time in seconds: how often the time offset has to be refreshed

      - ``indcash`` (default: ``True``)

        Manage IND codes as if they were cash for price retrieval
    """
    REQIDBASE = 16777216
    BrokerCls = None
    DataCls = None
    params = (('host', '127.0.0.1'), ('port', 7496), ('clientId', None), ('notifyall', False), ('_debug', False), ('reconnect', 3), ('timeout', 3.0), ('timeoffset', True), ('timerefresh', 60.0), ('indcash', True))

    @classmethod
    def getdata(cls, *args, **kwargs):
        if False:
            return 10
        'Returns ``DataCls`` with args, kwargs'
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns broker with *args, **kwargs from registered ``BrokerCls``'
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(IBStore, self).__init__()
        self._lock_q = threading.Lock()
        self._lock_accupd = threading.Lock()
        self._lock_pos = threading.Lock()
        self._lock_notif = threading.Lock()
        self._event_managed_accounts = threading.Event()
        self._event_accdownload = threading.Event()
        self.dontreconnect = False
        self._env = None
        self.broker = None
        self.datas = list()
        self.ccount = 0
        self._lock_tmoffset = threading.Lock()
        self.tmoffset = timedelta()
        self.qs = collections.OrderedDict()
        self.ts = collections.OrderedDict()
        self.iscash = dict()
        self.histexreq = dict()
        self.histfmt = dict()
        self.histsend = dict()
        self.histtz = dict()
        self.acc_cash = AutoDict()
        self.acc_value = AutoDict()
        self.acc_upds = AutoDict()
        self.port_update = False
        self.positions = collections.defaultdict(Position)
        self._tickerId = itertools.count(self.REQIDBASE)
        self.orderid = None
        self.cdetails = collections.defaultdict(list)
        self.managed_accounts = list()
        self.notifs = queue.Queue()
        if self.p.clientId is None:
            self.clientId = random.randint(1, pow(2, 16) - 1)
        else:
            self.clientId = self.p.clientId
        self.conn = ibopt.ibConnection(host=self.p.host, port=self.p.port, clientId=self.clientId)
        if self.p._debug or self.p.notifyall:
            self.conn.registerAll(self.watcher)
        methods = inspect.getmembers(self, inspect.ismethod)
        for (name, method) in methods:
            if not getattr(method, '_ibregister', False):
                continue
            message = getattr(ibopt.message, name)
            self.conn.register(method, message)

        def keyfn(x):
            if False:
                while True:
                    i = 10
            (n, t) = x.split()
            (tf, comp) = self._sizes[t]
            return (tf, int(n) * comp)

        def key2fn(x):
            if False:
                return 10
            (n, d) = x.split()
            tf = self._dur2tf[d]
            return (tf, int(n))
        self.revdur = collections.defaultdict(list)
        for (duration, barsizes) in self._durations.items():
            for barsize in barsizes:
                self.revdur[keyfn(barsize)].append(duration)
        for barsize in self.revdur:
            self.revdur[barsize].sort(key=key2fn)

    def start(self, data=None, broker=None):
        if False:
            print('Hello World!')
        self.reconnect(fromstart=True)
        if data is not None:
            self._env = data._env
            self.datas.append(data)
            return self.getTickerQueue(start=True)
        elif broker is not None:
            self.broker = broker

    def stop(self):
        if False:
            while True:
                i = 10
        try:
            self.conn.disconnect()
        except AttributeError:
            pass
        self._event_managed_accounts.set()
        self._event_accdownload.set()

    def logmsg(self, *args):
        if False:
            return 10
        if self.p._debug:
            print(*args)

    def watcher(self, msg):
        if False:
            print('Hello World!')
        self.logmsg(str(msg))
        if self.p.notifyall:
            self.notifs.put((msg, tuple(msg.values()), dict(msg.items())))

    def connected(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.conn.isConnected()
        except AttributeError:
            pass
        return False

    def reconnect(self, fromstart=False, resub=False):
        if False:
            for i in range(10):
                print('nop')
        firstconnect = False
        try:
            if self.conn.isConnected():
                if resub:
                    self.startdatas()
                return True
        except AttributeError:
            firstconnect = True
        if self.dontreconnect:
            return False
        retries = self.p.reconnect
        if retries >= 0:
            retries += firstconnect
        while retries < 0 or retries:
            if not firstconnect:
                time.sleep(self.p.timeout)
            firstconnect = False
            if self.conn.connect():
                if not fromstart or resub:
                    self.startdatas()
                return True
            if retries > 0:
                retries -= 1
        self.dontreconnect = True
        return False

    def startdatas(self):
        if False:
            print('Hello World!')
        ts = list()
        for data in self.datas:
            t = threading.Thread(target=data.reqdata)
            t.start()
            ts.append(t)
        for t in ts:
            t.join()

    def stopdatas(self):
        if False:
            return 10
        qs = list(self.qs.values())
        ts = list()
        for data in self.datas:
            t = threading.Thread(target=data.canceldata)
            t.start()
            ts.append(t)
        for t in ts:
            t.join()
        for q in reversed(qs):
            q.put(None)

    def get_notifications(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the pending "store" notifications'
        self.notifs.put(None)
        notifs = list()
        while True:
            notif = self.notifs.get()
            if notif is None:
                break
            notifs.append(notif)
        return notifs

    @ibregister
    def error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if not self.p.notifyall:
            self.notifs.put((msg, tuple(msg.values()), dict(msg.items())))
        if msg.errorCode is None:
            pass
        elif msg.errorCode in [200, 203, 162, 320, 321, 322]:
            try:
                q = self.qs[msg.id]
            except KeyError:
                pass
            else:
                self.cancelQueue(q, True)
        elif msg.errorCode in [354, 420]:
            try:
                q = self.qs[msg.id]
            except KeyError:
                pass
            else:
                q.put(-msg.errorCode)
                self.cancelQueue(q)
        elif msg.errorCode == 10225:
            try:
                q = self.qs[msg.id]
            except KeyError:
                pass
            else:
                q.put(-msg.errorCode)
        elif msg.errorCode == 326:
            self.dontreconnect = True
            self.conn.disconnect()
            self.stopdatas()
        elif msg.errorCode == 502:
            self.conn.disconnect()
            self.stopdatas()
        elif msg.errorCode == 504:
            pass
        elif msg.errorCode == 1300:
            self.conn.disconnect()
            self.stopdatas()
        elif msg.errorCode == 1100:
            for q in self.ts:
                q.put(-msg.errorCode)
        elif msg.errorCode == 1101:
            for q in self.ts:
                q.put(-msg.errorCode)
        elif msg.errorCode == 1102:
            for q in self.ts:
                q.put(-msg.errorCode)
        elif msg.errorCode < 500:
            if msg.id < self.REQIDBASE:
                if self.broker is not None:
                    self.broker.push_ordererror(msg)
            else:
                q = self.qs[msg.id]
                self.cancelQueue(q, True)

    @ibregister
    def connectionClosed(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.conn.disconnect()
        self.stopdatas()

    @ibregister
    def managedAccounts(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.managed_accounts = msg.accountsList.split(',')
        self._event_managed_accounts.set()
        self.reqCurrentTime()

    def reqCurrentTime(self):
        if False:
            while True:
                i = 10
        self.conn.reqCurrentTime()

    @ibregister
    def currentTime(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if not self.p.timeoffset:
            return
        curtime = datetime.fromtimestamp(float(msg.time))
        with self._lock_tmoffset:
            self.tmoffset = curtime - datetime.now()
        threading.Timer(self.p.timerefresh, self.reqCurrentTime).start()

    def timeoffset(self):
        if False:
            return 10
        with self._lock_tmoffset:
            return self.tmoffset

    def nextTickerId(self):
        if False:
            i = 10
            return i + 15
        return next(self._tickerId)

    @ibregister
    def nextValidId(self, msg):
        if False:
            return 10
        self.orderid = itertools.count(msg.orderId)

    def nextOrderId(self):
        if False:
            print('Hello World!')
        return next(self.orderid)

    def reuseQueue(self, tickerId):
        if False:
            while True:
                i = 10
        'Reuses queue for tickerId, returning the new tickerId and q'
        with self._lock_q:
            q = self.qs.pop(tickerId, None)
            iscash = self.iscash.pop(tickerId, None)
            tickerId = self.nextTickerId()
            self.ts[q] = tickerId
            self.qs[tickerId] = q
            self.iscash[tickerId] = iscash
        return (tickerId, q)

    def getTickerQueue(self, start=False):
        if False:
            while True:
                i = 10
        'Creates ticker/Queue for data delivery to a data feed'
        q = queue.Queue()
        if start:
            q.put(None)
            return q
        with self._lock_q:
            tickerId = self.nextTickerId()
            self.qs[tickerId] = q
            self.ts[q] = tickerId
            self.iscash[tickerId] = False
        return (tickerId, q)

    def cancelQueue(self, q, sendnone=False):
        if False:
            i = 10
            return i + 15
        'Cancels a Queue for data delivery'
        tickerId = self.ts.pop(q, None)
        self.qs.pop(tickerId, None)
        self.iscash.pop(tickerId, None)
        if sendnone:
            q.put(None)

    def validQueue(self, q):
        if False:
            for i in range(10):
                print('nop')
        'Returns (bool)  if a queue is still valid'
        return q in self.ts

    def getContractDetails(self, contract, maxcount=None):
        if False:
            while True:
                i = 10
        cds = list()
        q = self.reqContractDetails(contract)
        while True:
            msg = q.get()
            if msg is None:
                break
            cds.append(msg)
        if not cds or (maxcount and len(cds) > maxcount):
            err = 'Ambiguous contract: none/multiple answers received'
            self.notifs.put((err, cds, {}))
            return None
        return cds

    def reqContractDetails(self, contract):
        if False:
            print('Hello World!')
        (tickerId, q) = self.getTickerQueue()
        self.conn.reqContractDetails(tickerId, contract)
        return q

    @ibregister
    def contractDetailsEnd(self, msg):
        if False:
            while True:
                i = 10
        'Signal end of contractdetails'
        self.cancelQueue(self.qs[msg.reqId], True)

    @ibregister
    def contractDetails(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Receive answer and pass it to the queue'
        self.qs[msg.reqId].put(msg)

    def reqHistoricalDataEx(self, contract, enddate, begindate, timeframe, compression, what=None, useRTH=False, tz='', sessionend=None, tickerId=None):
        if False:
            i = 10
            return i + 15
        '\n        Extension of the raw reqHistoricalData proxy, which takes two dates\n        rather than a duration, barsize and date\n\n        It uses the IB published valid duration/barsizes to make a mapping and\n        spread a historical request over several historical requests if needed\n        '
        kwargs = locals().copy()
        kwargs.pop('self', None)
        if timeframe < TimeFrame.Seconds:
            return self.getTickerQueue(start=True)
        if enddate is None:
            enddate = datetime.now()
        if begindate is None:
            duration = self.getmaxduration(timeframe, compression)
            if duration is None:
                err = 'No duration for historical data request for timeframe/compresison'
                self.notifs.put((err, (), kwargs))
                return self.getTickerQueue(start=True)
            barsize = self.tfcomp_to_size(timeframe, compression)
            if barsize is None:
                err = 'No supported barsize for historical data request for timeframe/compresison'
                self.notifs.put((err, (), kwargs))
                return self.getTickerQueue(start=True)
            return self.reqHistoricalData(contract=contract, enddate=enddate, duration=duration, barsize=barsize, what=what, useRTH=useRTH, tz=tz, sessionend=sessionend)
        durations = self.getdurations(timeframe, compression)
        if not durations:
            return self.getTickerQueue(start=True)
        if tickerId is None:
            (tickerId, q) = self.getTickerQueue()
        else:
            (tickerId, q) = self.reuseQueue(tickerId)
        duration = None
        for dur in durations:
            intdate = self.dt_plus_duration(begindate, dur)
            if intdate >= enddate:
                intdate = enddate
                duration = dur
                break
        if duration is None:
            duration = durations[-1]
            self.histexreq[tickerId] = dict(contract=contract, enddate=enddate, begindate=intdate, timeframe=timeframe, compression=compression, what=what, useRTH=useRTH, tz=tz, sessionend=sessionend)
        barsize = self.tfcomp_to_size(timeframe, compression)
        self.histfmt[tickerId] = timeframe >= TimeFrame.Days
        self.histsend[tickerId] = sessionend
        self.histtz[tickerId] = tz
        if contract.m_secType in ['CASH', 'CFD']:
            self.iscash[tickerId] = 1
            if not what:
                what = 'BID'
        elif contract.m_secType in ['IND'] and self.p.indcash:
            self.iscash[tickerId] = 4
        what = what or 'TRADES'
        self.conn.reqHistoricalData(tickerId, contract, bytes(intdate.strftime('%Y%m%d %H:%M:%S') + ' GMT'), bytes(duration), bytes(barsize), bytes(what), int(useRTH), 2)
        return q

    def reqHistoricalData(self, contract, enddate, duration, barsize, what=None, useRTH=False, tz='', sessionend=None):
        if False:
            for i in range(10):
                print('nop')
        'Proxy to reqHistorical Data'
        (tickerId, q) = self.getTickerQueue()
        if contract.m_secType in ['CASH', 'CFD']:
            self.iscash[tickerId] = True
            if not what:
                what = 'BID'
            elif what == 'ASK':
                self.iscash[tickerId] = 2
        else:
            what = what or 'TRADES'
        tframe = self._sizes[barsize.split()[1]][0]
        self.histfmt[tickerId] = tframe >= TimeFrame.Days
        self.histsend[tickerId] = sessionend
        self.histtz[tickerId] = tz
        self.conn.reqHistoricalData(tickerId, contract, bytes(enddate.strftime('%Y%m%d %H:%M:%S') + ' GMT'), bytes(duration), bytes(barsize), bytes(what), int(useRTH), 2)
        return q

    def cancelHistoricalData(self, q):
        if False:
            return 10
        'Cancels an existing HistoricalData request\n\n        Params:\n          - q: the Queue returned by reqMktData\n        '
        with self._lock_q:
            self.conn.cancelHistoricalData(self.ts[q])
            self.cancelQueue(q, True)

    def reqRealTimeBars(self, contract, useRTH=False, duration=5):
        if False:
            i = 10
            return i + 15
        'Creates a request for (5 seconds) Real Time Bars\n\n        Params:\n          - contract: a ib.ext.Contract.Contract intance\n          - useRTH: (default: False) passed to TWS\n          - duration: (default: 5) passed to TWS, no other value works in 2016)\n\n        Returns:\n          - a Queue the client can wait on to receive a RTVolume instance\n        '
        (tickerId, q) = self.getTickerQueue()
        self.conn.reqRealTimeBars(tickerId, contract, duration, bytes('TRADES'), int(useRTH))
        return q

    def cancelRealTimeBars(self, q):
        if False:
            for i in range(10):
                print('nop')
        'Cancels an existing MarketData subscription\n\n        Params:\n          - q: the Queue returned by reqMktData\n        '
        with self._lock_q:
            tickerId = self.ts.get(q, None)
            if tickerId is not None:
                self.conn.cancelRealTimeBars(tickerId)
            self.cancelQueue(q, True)

    def reqMktData(self, contract, what=None):
        if False:
            print('Hello World!')
        'Creates a MarketData subscription\n\n        Params:\n          - contract: a ib.ext.Contract.Contract intance\n\n        Returns:\n          - a Queue the client can wait on to receive a RTVolume instance\n        '
        (tickerId, q) = self.getTickerQueue()
        ticks = '233'
        if contract.m_secType in ['CASH', 'CFD']:
            self.iscash[tickerId] = True
            ticks = ''
            if what == 'ASK':
                self.iscash[tickerId] = 2
        self.conn.reqMktData(tickerId, contract, bytes(ticks), False)
        return q

    def cancelMktData(self, q):
        if False:
            return 10
        'Cancels an existing MarketData subscription\n\n        Params:\n          - q: the Queue returned by reqMktData\n        '
        with self._lock_q:
            tickerId = self.ts.get(q, None)
            if tickerId is not None:
                self.conn.cancelMktData(tickerId)
            self.cancelQueue(q, True)

    @ibregister
    def tickString(self, msg):
        if False:
            return 10
        if msg.tickType == 48:
            try:
                rtvol = RTVolume(msg.value)
            except ValueError:
                pass
            else:
                self.qs[msg.tickerId].put(rtvol)

    @ibregister
    def tickPrice(self, msg):
        if False:
            return 10
        'Cash Markets have no notion of "last_price"/"last_size" and the\n        tracking of the price is done (industry de-facto standard at least with\n        the IB API) following the BID price\n\n        A RTVolume which will only contain a price is put into the client\'s\n        queue to have a consistent cross-market interface\n        '
        tickerId = msg.tickerId
        fieldcode = self.iscash[tickerId]
        if fieldcode:
            if msg.field == fieldcode:
                try:
                    if msg.price == -1.0:
                        return
                except AttributeError:
                    pass
                try:
                    rtvol = RTVolume(price=msg.price, tmoffset=self.tmoffset)
                except ValueError:
                    pass
                else:
                    self.qs[tickerId].put(rtvol)

    @ibregister
    def realtimeBar(self, msg):
        if False:
            return 10
        'Receives x seconds Real Time Bars (at the time of writing only 5\n        seconds are supported)\n\n        Not valid for cash markets\n        '
        msg.time = datetime.utcfromtimestamp(float(msg.time))
        self.qs[msg.reqId].put(msg)

    @ibregister
    def historicalData(self, msg):
        if False:
            while True:
                i = 10
        'Receives the events of a historical data request'
        tickerId = msg.reqId
        q = self.qs[tickerId]
        if msg.date.startswith('finished-'):
            self.histfmt.pop(tickerId, None)
            self.histsend.pop(tickerId, None)
            self.histtz.pop(tickerId, None)
            kargs = self.histexreq.pop(tickerId, None)
            if kargs is not None:
                self.reqHistoricalDataEx(tickerId=tickerId, **kargs)
                return
            msg.date = None
            self.cancelQueue(q)
        else:
            dtstr = msg.date
            if self.histfmt[tickerId]:
                sessionend = self.histsend[tickerId]
                dt = datetime.strptime(dtstr, '%Y%m%d')
                dteos = datetime.combine(dt, sessionend)
                tz = self.histtz[tickerId]
                if tz:
                    dteostz = tz.localize(dteos)
                    dteosutc = dteostz.astimezone(UTC).replace(tzinfo=None)
                else:
                    dteosutc = dteos
                if dteosutc <= datetime.utcnow():
                    dt = dteosutc
                msg.date = dt
            else:
                msg.date = datetime.utcfromtimestamp(long(dtstr))
        q.put(msg)
    _durations = dict([('60 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min')), ('120 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins')), ('180 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins')), ('300 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins')), ('600 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins')), ('900 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins')), ('1200 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins')), ('1800 S', ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins')), ('3600 S', ('5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour')), ('7200 S', ('5 secs', '10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours')), ('10800 S', ('10 secs', '15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours')), ('14400 S', ('15 secs', '30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours')), ('28800 S', ('30 secs', '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours')), ('1 D', ('1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day')), ('2 D', ('2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day')), ('1 W', ('3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day', '1 W')), ('2 W', ('15 mins', '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day', '1 W')), ('1 M', ('30 mins', '1 hour', '2 hours', '3 hours', '4 hours', '8 hours', '1 day', '1 W', '1 M')), ('2 M', ('1 day', '1 W', '1 M')), ('3 M', ('1 day', '1 W', '1 M')), ('4 M', ('1 day', '1 W', '1 M')), ('5 M', ('1 day', '1 W', '1 M')), ('6 M', ('1 day', '1 W', '1 M')), ('7 M', ('1 day', '1 W', '1 M')), ('8 M', ('1 day', '1 W', '1 M')), ('9 M', ('1 day', '1 W', '1 M')), ('10 M', ('1 day', '1 W', '1 M')), ('11 M', ('1 day', '1 W', '1 M')), ('1 Y', ('1 day', '1 W', '1 M'))])
    _sizes = {'secs': (TimeFrame.Seconds, 1), 'min': (TimeFrame.Minutes, 1), 'mins': (TimeFrame.Minutes, 1), 'hour': (TimeFrame.Minutes, 60), 'hours': (TimeFrame.Minutes, 60), 'day': (TimeFrame.Days, 1), 'W': (TimeFrame.Weeks, 1), 'M': (TimeFrame.Months, 1)}
    _dur2tf = {'S': TimeFrame.Seconds, 'D': TimeFrame.Days, 'W': TimeFrame.Weeks, 'M': TimeFrame.Months, 'Y': TimeFrame.Years}

    def getdurations(self, timeframe, compression):
        if False:
            return 10
        key = (timeframe, compression)
        if key not in self.revdur:
            return []
        return self.revdur[key]

    def getmaxduration(self, timeframe, compression):
        if False:
            while True:
                i = 10
        key = (timeframe, compression)
        try:
            return self.revdur[key][-1]
        except (KeyError, IndexError):
            pass
        return None

    def tfcomp_to_size(self, timeframe, compression):
        if False:
            for i in range(10):
                print('nop')
        if timeframe == TimeFrame.Months:
            return '{} M'.format(compression)
        if timeframe == TimeFrame.Weeks:
            return '{} W'.format(compression)
        if timeframe == TimeFrame.Days:
            if not compression % 7:
                return '{} W'.format(compression // 7)
            return '{} day'.format(compression)
        if timeframe == TimeFrame.Minutes:
            if not compression % 60:
                hours = compression // 60
                return '{} hour'.format(hours) + 's' * (hours > 1)
            return '{} min'.format(compression) + 's' * (compression > 1)
        if timeframe == TimeFrame.Seconds:
            return '{} secs'.format(compression)
        return None

    def dt_plus_duration(self, dt, duration):
        if False:
            print('Hello World!')
        (size, dim) = duration.split()
        size = int(size)
        if dim == 'S':
            return dt + timedelta(seconds=size)
        if dim == 'D':
            return dt + timedelta(days=size)
        if dim == 'W':
            return dt + timedelta(days=size * 7)
        if dim == 'M':
            month = dt.month - 1 + size
            (years, month) = divmod(month, 12)
            return dt.replace(year=dt.year + years, month=month + 1)
        if dim == 'Y':
            return dt.replace(year=dt.year + size)
        return dt

    def calcdurations(self, dtbegin, dtend):
        if False:
            i = 10
            return i + 15
        'Calculate a duration in between 2 datetimes'
        duration = self.histduration(dtbegin, dtend)
        if duration[-1] == 'M':
            m = int(duration.split()[0])
            m1 = min(2, m)
            m2 = max(1, m1)
            checkdur = '{} M'.format(m2)
        elif duration[-1] == 'Y':
            checkdur = '1 Y'
        else:
            checkdur = duration
        sizes = self._durations[checkduration]
        return (duration, sizes)

    def calcduration(self, dtbegin, dtend):
        if False:
            return 10
        'Calculate a duration in between 2 datetimes. Returns single size'
        (duration, sizes) = self._calcdurations(dtbegin, dtend)
        return (duration, sizes[0])

    def histduration(self, dt1, dt2):
        if False:
            while True:
                i = 10
        td = dt2 - dt1
        tsecs = td.total_seconds()
        secs = [60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400, 28800]
        idxsec = bisect.bisect_left(secs, tsecs)
        if idxsec < len(secs):
            return '{} S'.format(secs[idxsec])
        tdextra = bool(td.seconds or td.microseconds)
        days = td.days + tdextra
        if td.days <= 2:
            return '{} D'.format(days)
        (weeks, d) = divmod(td.days, 7)
        weeks += bool(d or tdextra)
        if weeks <= 2:
            return '{} W'.format(weeks)
        (y2, m2, d2) = (dt2.year, dt2.month, dt2.day)
        (y1, m1, d1) = (dt1.year, dt1.month, dt2.day)
        (H2, M2, S2, US2) = (dt2.hour, dt2.minute, dt2.second, dt2.microsecond)
        (H1, M1, S1, US1) = (dt1.hour, dt1.minute, dt1.second, dt1.microsecond)
        months = y2 * 12 + m2 - (y1 * 12 + m1) + ((d2, H2, M2, S2, US2) > (d1, H1, M1, S1, US1))
        if months <= 1:
            return '1 M'
        elif months <= 11:
            return '2 M'
        return '1 Y'

    def makecontract(self, symbol, sectype, exch, curr, expiry='', strike=0.0, right='', mult=1):
        if False:
            return 10
        'returns a contract from the parameters without check'
        contract = Contract()
        contract.m_symbol = bytes(symbol)
        contract.m_secType = bytes(sectype)
        contract.m_exchange = bytes(exch)
        if curr:
            contract.m_currency = bytes(curr)
        if sectype in ['FUT', 'OPT', 'FOP']:
            contract.m_expiry = bytes(expiry)
        if sectype in ['OPT', 'FOP']:
            contract.m_strike = strike
            contract.m_right = bytes(right)
        if mult:
            contract.m_multiplier = bytes(mult)
        return contract

    def cancelOrder(self, orderid):
        if False:
            i = 10
            return i + 15
        'Proxy to cancelOrder'
        self.conn.cancelOrder(orderid)

    def placeOrder(self, orderid, contract, order):
        if False:
            print('Hello World!')
        'Proxy to placeOrder'
        self.conn.placeOrder(orderid, contract, order)

    @ibregister
    def openOrder(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Receive the event ``openOrder`` events'
        self.broker.push_orderstate(msg)

    @ibregister
    def execDetails(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Receive execDetails'
        self.broker.push_execution(msg.execution)

    @ibregister
    def orderStatus(self, msg):
        if False:
            return 10
        'Receive the event ``orderStatus``'
        self.broker.push_orderstatus(msg)

    @ibregister
    def commissionReport(self, msg):
        if False:
            while True:
                i = 10
        'Receive the event commissionReport'
        self.broker.push_commissionreport(msg.commissionReport)

    def reqPositions(self):
        if False:
            return 10
        'Proxy to reqPositions'
        self.conn.reqPositions()

    @ibregister
    def position(self, msg):
        if False:
            i = 10
            return i + 15
        'Receive event positions'
        pass

    def reqAccountUpdates(self, subscribe=True, account=None):
        if False:
            print('Hello World!')
        'Proxy to reqAccountUpdates\n\n        If ``account`` is ``None``, wait for the ``managedAccounts`` message to\n        set the account codes\n        '
        if account is None:
            self._event_managed_accounts.wait()
            account = self.managed_accounts[0]
        self.conn.reqAccountUpdates(subscribe, bytes(account))

    @ibregister
    def accountDownloadEnd(self, msg):
        if False:
            i = 10
            return i + 15
        self._event_accdownload.set()
        if False:
            if self.port_update:
                self.broker.push_portupdate()
                self.port_update = False

    @ibregister
    def updatePortfolio(self, msg):
        if False:
            for i in range(10):
                print('nop')
        with self._lock_pos:
            if not self._event_accdownload.is_set():
                position = Position(msg.position, msg.averageCost)
                self.positions[msg.contract.m_conId] = position
            else:
                position = self.positions[msg.contract.m_conId]
                if not position.fix(msg.position, msg.averageCost):
                    err = 'The current calculated position and the position reported by the broker do not match. Operation can continue, but the trades calculated in the strategy may be wrong'
                    self.notifs.put((err, (), {}))
                self.broker.push_portupdate()

    def getposition(self, contract, clone=False):
        if False:
            while True:
                i = 10
        with self._lock_pos:
            position = self.positions[contract.m_conId]
            if clone:
                return copy(position)
            return position

    @ibregister
    def updateAccountValue(self, msg):
        if False:
            for i in range(10):
                print('nop')
        with self._lock_accupd:
            try:
                value = float(msg.value)
            except ValueError:
                value = msg.value
            self.acc_upds[msg.accountName][msg.key][msg.currency] = value
            if msg.key == 'NetLiquidation':
                self.acc_value[msg.accountName] = value
            elif msg.key == 'TotalCashBalance' and msg.currency == 'BASE':
                self.acc_cash[msg.accountName] = value

    def get_acc_values(self, account=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns all account value infos sent by TWS during regular updates\n        Waits for at least 1 successful download\n\n        If ``account`` is ``None`` then a dictionary with accounts as keys will\n        be returned containing all accounts\n\n        If account is specified or the system has only 1 account the dictionary\n        corresponding to that account is returned\n        '
        if self.connected():
            self._event_accdownload.wait()
        with self._updacclock:
            if account is None:
                if self.connected():
                    self._event_managed_accounts.wait()
                if not self.managed_accounts:
                    return self.acc_upds.copy()
                elif len(self.managed_accounts) > 1:
                    return self.acc_upds.copy()
                account = self.managed_accounts[0]
            try:
                return self.acc_upds[account].copy()
            except KeyError:
                pass
            return self.acc_upds.copy()

    def get_acc_value(self, account=None):
        if False:
            i = 10
            return i + 15
        'Returns the net liquidation value sent by TWS during regular updates\n        Waits for at least 1 successful download\n\n        If ``account`` is ``None`` then a dictionary with accounts as keys will\n        be returned containing all accounts\n\n        If account is specified or the system has only 1 account the dictionary\n        corresponding to that account is returned\n        '
        if self.connected():
            self._event_accdownload.wait()
        with self._lock_accupd:
            if account is None:
                if self.connected():
                    self._event_managed_accounts.wait()
                if not self.managed_accounts:
                    return float()
                elif len(self.managed_accounts) > 1:
                    return sum(self.acc_value.values())
                account = self.managed_accounts[0]
            try:
                return self.acc_value[account]
            except KeyError:
                pass
            return float()

    def get_acc_cash(self, account=None):
        if False:
            return 10
        'Returns the total cash value sent by TWS during regular updates\n        Waits for at least 1 successful download\n\n        If ``account`` is ``None`` then a dictionary with accounts as keys will\n        be returned containing all accounts\n\n        If account is specified or the system has only 1 account the dictionary\n        corresponding to that account is returned\n        '
        if self.connected():
            self._event_accdownload.wait()
        with self._lock_accupd:
            if account is None:
                if self.connected():
                    self._event_managed_accounts.wait()
                if not self.managed_accounts:
                    return float()
                elif len(self.managed_accounts) > 1:
                    return sum(self.acc_cash.values())
                account = self.managed_accounts[0]
            try:
                return self.acc_cash[account]
            except KeyError:
                pass
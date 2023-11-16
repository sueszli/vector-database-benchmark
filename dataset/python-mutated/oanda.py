from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, timedelta
from backtrader.feed import DataBase
from backtrader import TimeFrame, date2num, num2date
from backtrader.utils.py3 import integer_types, queue, string_types, with_metaclass
from backtrader.metabase import MetaParams
from backtrader.stores import oandastore

class MetaOandaData(DataBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        'Class has already been created ... register'
        super(MetaOandaData, cls).__init__(name, bases, dct)
        oandastore.OandaStore.DataCls = cls

class OandaData(with_metaclass(MetaOandaData, DataBase)):
    """Oanda Data Feed.

    Params:

      - ``qcheck`` (default: ``0.5``)

        Time in seconds to wake up if no data is received to give a chance to
        resample/replay packets properly and pass notifications up the chain

      - ``historical`` (default: ``False``)

        If set to ``True`` the data feed will stop after doing the first
        download of data.

        The standard data feed parameters ``fromdate`` and ``todate`` will be
        used as reference.

        The data feed will make multiple requests if the requested duration is
        larger than the one allowed by IB given the timeframe/compression
        chosen for the data.

      - ``backfill_start`` (default: ``True``)

        Perform backfilling at the start. The maximum possible historical data
        will be fetched in a single request.

      - ``backfill`` (default: ``True``)

        Perform backfilling after a disconnection/reconnection cycle. The gap
        duration will be used to download the smallest possible amount of data

      - ``backfill_from`` (default: ``None``)

        An additional data source can be passed to do an initial layer of
        backfilling. Once the data source is depleted and if requested,
        backfilling from IB will take place. This is ideally meant to backfill
        from already stored sources like a file on disk, but not limited to.

      - ``bidask`` (default: ``True``)

        If ``True``, then the historical/backfilling requests will request
        bid/ask prices from the server

        If ``False``, then *midpoint* will be requested

      - ``useask`` (default: ``False``)

        If ``True`` the *ask* part of the *bidask* prices will be used instead
        of the default use of *bid*

      - ``includeFirst`` (default: ``True``)

        Influence the delivery of the 1st bar of a historical/backfilling
        request by setting the parameter directly to the Oanda API calls

      - ``reconnect`` (default: ``True``)

        Reconnect when network connection is down

      - ``reconnections`` (default: ``-1``)

        Number of times to attempt reconnections: ``-1`` means forever

      - ``reconntimeout`` (default: ``5.0``)

        Time in seconds to wait in between reconnection attemps

    This data feed supports only this mapping of ``timeframe`` and
    ``compression``, which comply with the definitions in the OANDA API
    Developer's Guid::

        (TimeFrame.Seconds, 5): 'S5',
        (TimeFrame.Seconds, 10): 'S10',
        (TimeFrame.Seconds, 15): 'S15',
        (TimeFrame.Seconds, 30): 'S30',
        (TimeFrame.Minutes, 1): 'M1',
        (TimeFrame.Minutes, 2): 'M3',
        (TimeFrame.Minutes, 3): 'M3',
        (TimeFrame.Minutes, 4): 'M4',
        (TimeFrame.Minutes, 5): 'M5',
        (TimeFrame.Minutes, 10): 'M10',
        (TimeFrame.Minutes, 15): 'M15',
        (TimeFrame.Minutes, 30): 'M30',
        (TimeFrame.Minutes, 60): 'H1',
        (TimeFrame.Minutes, 120): 'H2',
        (TimeFrame.Minutes, 180): 'H3',
        (TimeFrame.Minutes, 240): 'H4',
        (TimeFrame.Minutes, 360): 'H6',
        (TimeFrame.Minutes, 480): 'H8',
        (TimeFrame.Days, 1): 'D',
        (TimeFrame.Weeks, 1): 'W',
        (TimeFrame.Months, 1): 'M',

    Any other combination will be rejected
    """
    params = (('qcheck', 0.5), ('historical', False), ('backfill_start', True), ('backfill', True), ('backfill_from', None), ('bidask', True), ('useask', False), ('includeFirst', True), ('reconnect', True), ('reconnections', -1), ('reconntimeout', 5.0))
    _store = oandastore.OandaStore
    (_ST_FROM, _ST_START, _ST_LIVE, _ST_HISTORBACK, _ST_OVER) = range(5)
    _TOFFSET = timedelta()

    def _timeoffset(self):
        if False:
            while True:
                i = 10
        return self._TOFFSET

    def islive(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns ``True`` to notify ``Cerebro`` that preloading and runonce\n        should be deactivated'
        return True

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.o = self._store(**kwargs)
        self._candleFormat = 'bidask' if self.p.bidask else 'midpoint'

    def setenvironment(self, env):
        if False:
            print('Hello World!')
        'Receives an environment (cerebro) and passes it over to the store it\n        belongs to'
        super(OandaData, self).setenvironment(env)
        env.addstore(self.o)

    def start(self):
        if False:
            print('Hello World!')
        'Starts the Oanda connecction and gets the real contract and\n        contractdetails if it exists'
        super(OandaData, self).start()
        self._statelivereconn = False
        self._storedmsg = dict()
        self.qlive = queue.Queue()
        self._state = self._ST_OVER
        self.o.start(data=self)
        otf = self.o.get_granularity(self._timeframe, self._compression)
        if otf is None:
            self.put_notification(self.NOTSUPPORTED_TF)
            self._state = self._ST_OVER
            return
        self.contractdetails = cd = self.o.get_instrument(self.p.dataname)
        if cd is None:
            self.put_notification(self.NOTSUBSCRIBED)
            self._state = self._ST_OVER
            return
        if self.p.backfill_from is not None:
            self._state = self._ST_FROM
            self.p.backfill_from._start()
        else:
            self._start_finish()
            self._state = self._ST_START
            self._st_start()
        self._reconns = 0

    def _st_start(self, instart=True, tmout=None):
        if False:
            return 10
        if self.p.historical:
            self.put_notification(self.DELAYED)
            dtend = None
            if self.todate < float('inf'):
                dtend = num2date(self.todate)
            dtbegin = None
            if self.fromdate > float('-inf'):
                dtbegin = num2date(self.fromdate)
            self.qhist = self.o.candles(self.p.dataname, dtbegin, dtend, self._timeframe, self._compression, candleFormat=self._candleFormat, includeFirst=self.p.includeFirst)
            self._state = self._ST_HISTORBACK
            return True
        self.qlive = self.o.streaming_prices(self.p.dataname, tmout=tmout)
        if instart:
            self._statelivereconn = self.p.backfill_start
        else:
            self._statelivereconn = self.p.backfill
        if self._statelivereconn:
            self.put_notification(self.DELAYED)
        self._state = self._ST_LIVE
        if instart:
            self._reconns = self.p.reconnections
        return True

    def stop(self):
        if False:
            print('Hello World!')
        'Stops and tells the store to stop'
        super(OandaData, self).stop()
        self.o.stop()

    def haslivedata(self):
        if False:
            i = 10
            return i + 15
        return bool(self._storedmsg or self.qlive)

    def _load(self):
        if False:
            for i in range(10):
                print('nop')
        if self._state == self._ST_OVER:
            return False
        while True:
            if self._state == self._ST_LIVE:
                try:
                    msg = self._storedmsg.pop(None, None) or self.qlive.get(timeout=self._qcheck)
                except queue.Empty:
                    return None
                if msg is None:
                    self.put_notification(self.CONNBROKEN)
                    if not self.p.reconnect or self._reconns == 0:
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False
                    self._reconns -= 1
                    self._st_start(instart=False, tmout=self.p.reconntimeout)
                    continue
                if 'code' in msg:
                    self.put_notification(self.CONNBROKEN)
                    code = msg['code']
                    if code not in [599, 598, 596]:
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False
                    if not self.p.reconnect or self._reconns == 0:
                        self.put_notification(self.DISCONNECTED)
                        self._state = self._ST_OVER
                        return False
                    self._reconns -= 1
                    self._st_start(instart=False, tmout=self.p.reconntimeout)
                    continue
                self._reconns = self.p.reconnections
                if not self._statelivereconn:
                    if self._laststatus != self.LIVE:
                        if self.qlive.qsize() <= 1:
                            self.put_notification(self.LIVE)
                    ret = self._load_tick(msg)
                    if ret:
                        return True
                    continue
                self._storedmsg[None] = msg
                if self._laststatus != self.DELAYED:
                    self.put_notification(self.DELAYED)
                dtend = None
                if len(self) > 1:
                    dtbegin = self.datetime.datetime(-1)
                elif self.fromdate > float('-inf'):
                    dtbegin = num2date(self.fromdate)
                else:
                    dtbegin = None
                dtend = datetime.utcfromtimestamp(int(msg['time']) / 10 ** 6)
                self.qhist = self.o.candles(self.p.dataname, dtbegin, dtend, self._timeframe, self._compression, candleFormat=self._candleFormat, includeFirst=self.p.includeFirst)
                self._state = self._ST_HISTORBACK
                self._statelivereconn = False
                continue
            elif self._state == self._ST_HISTORBACK:
                msg = self.qhist.get()
                if msg is None:
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False
                elif 'code' in msg:
                    self.put_notification(self.NOTSUBSCRIBED)
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False
                if msg:
                    if self._load_history(msg):
                        return True
                    continue
                elif self.p.historical:
                    self.put_notification(self.DISCONNECTED)
                    self._state = self._ST_OVER
                    return False
                self._state = self._ST_LIVE
                continue
            elif self._state == self._ST_FROM:
                if not self.p.backfill_from.next():
                    self._state = self._ST_START
                    continue
                for alias in self.lines.getlinealiases():
                    lsrc = getattr(self.p.backfill_from.lines, alias)
                    ldst = getattr(self.lines, alias)
                    ldst[0] = lsrc[0]
                return True
            elif self._state == self._ST_START:
                if not self._st_start(instart=False):
                    self._state = self._ST_OVER
                    return False

    def _load_tick(self, msg):
        if False:
            return 10
        dtobj = datetime.utcfromtimestamp(int(msg['time']) / 10 ** 6)
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False
        self.lines.datetime[0] = dt
        self.lines.volume[0] = 0.0
        self.lines.openinterest[0] = 0.0
        tick = float(msg['ask']) if self.p.useask else float(msg['bid'])
        self.lines.open[0] = tick
        self.lines.high[0] = tick
        self.lines.low[0] = tick
        self.lines.close[0] = tick
        self.lines.volume[0] = 0.0
        self.lines.openinterest[0] = 0.0
        return True

    def _load_history(self, msg):
        if False:
            while True:
                i = 10
        dtobj = datetime.utcfromtimestamp(int(msg['time']) / 10 ** 6)
        dt = date2num(dtobj)
        if dt <= self.lines.datetime[-1]:
            return False
        self.lines.datetime[0] = dt
        self.lines.volume[0] = float(msg['volume'])
        self.lines.openinterest[0] = 0.0
        if self.p.bidask:
            if not self.p.useask:
                self.lines.open[0] = float(msg['openBid'])
                self.lines.high[0] = float(msg['highBid'])
                self.lines.low[0] = float(msg['lowBid'])
                self.lines.close[0] = float(msg['closeBid'])
            else:
                self.lines.open[0] = float(msg['openAsk'])
                self.lines.high[0] = float(msg['highAsk'])
                self.lines.low[0] = float(msg['lowAsk'])
                self.lines.close[0] = float(msg['closeAsk'])
        else:
            self.lines.open[0] = float(msg['openMid'])
            self.lines.high[0] = float(msg['highMid'])
            self.lines.low[0] = float(msg['lowMid'])
            self.lines.close[0] = float(msg['closeMid'])
        return True
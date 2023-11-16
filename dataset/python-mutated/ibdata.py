from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import backtrader as bt
from backtrader.feed import DataBase
from backtrader import TimeFrame, date2num, num2date
from backtrader.utils.py3 import integer_types, queue, string_types, with_metaclass
from backtrader.metabase import MetaParams
from backtrader.stores import ibstore

class MetaIBData(DataBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            return 10
        'Class has already been created ... register'
        super(MetaIBData, cls).__init__(name, bases, dct)
        ibstore.IBStore.DataCls = cls

class IBData(with_metaclass(MetaIBData, DataBase)):
    """Interactive Brokers Data Feed.

    Supports the following contract specifications in parameter ``dataname``:

          - TICKER  # Stock type and SMART exchange
          - TICKER-STK  # Stock and SMART exchange
          - TICKER-STK-EXCHANGE  # Stock
          - TICKER-STK-EXCHANGE-CURRENCY  # Stock

          - TICKER-CFD  # CFD and SMART exchange
          - TICKER-CFD-EXCHANGE  # CFD
          - TICKER-CDF-EXCHANGE-CURRENCY  # Stock

          - TICKER-IND-EXCHANGE  # Index
          - TICKER-IND-EXCHANGE-CURRENCY  # Index

          - TICKER-YYYYMM-EXCHANGE  # Future
          - TICKER-YYYYMM-EXCHANGE-CURRENCY  # Future
          - TICKER-YYYYMM-EXCHANGE-CURRENCY-MULT  # Future
          - TICKER-FUT-EXCHANGE-CURRENCY-YYYYMM-MULT # Future

          - TICKER-YYYYMM-EXCHANGE-CURRENCY-STRIKE-RIGHT  # FOP
          - TICKER-YYYYMM-EXCHANGE-CURRENCY-STRIKE-RIGHT-MULT  # FOP
          - TICKER-FOP-EXCHANGE-CURRENCY-YYYYMM-STRIKE-RIGHT # FOP
          - TICKER-FOP-EXCHANGE-CURRENCY-YYYYMM-STRIKE-RIGHT-MULT # FOP

          - CUR1.CUR2-CASH-IDEALPRO  # Forex

          - TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT  # OPT
          - TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT-MULT  # OPT
          - TICKER-OPT-EXCHANGE-CURRENCY-YYYYMMDD-STRIKE-RIGHT # OPT
          - TICKER-OPT-EXCHANGE-CURRENCY-YYYYMMDD-STRIKE-RIGHT-MULT # OPT

    Params:

      - ``sectype`` (default: ``STK``)

        Default value to apply as *security type* if not provided in the
        ``dataname`` specification

      - ``exchange`` (default: ``SMART``)

        Default value to apply as *exchange* if not provided in the
        ``dataname`` specification

      - ``currency`` (default: ``''``)

        Default value to apply as *currency* if not provided in the
        ``dataname`` specification

      - ``historical`` (default: ``False``)

        If set to ``True`` the data feed will stop after doing the first
        download of data.

        The standard data feed parameters ``fromdate`` and ``todate`` will be
        used as reference.

        The data feed will make multiple requests if the requested duration is
        larger than the one allowed by IB given the timeframe/compression
        chosen for the data.

      - ``what`` (default: ``None``)

        If ``None`` the default for different assets types will be used for
        historical data requests:

          - 'BID' for CASH assets
          - 'TRADES' for any other

        Use 'ASK' for the Ask quote of cash assets
        
        Check the IB API docs if another value is wished

      - ``rtbar`` (default: ``False``)

        If ``True`` the ``5 Seconds Realtime bars`` provided by Interactive
        Brokers will be used as the smalles tick. According to the
        documentation they correspond to real-time values (once collated and
        curated by IB)

        If ``False`` then the ``RTVolume`` prices will be used, which are based
        on receiving ticks. In the case of ``CASH`` assets (like for example
        EUR.JPY) ``RTVolume`` will always be used and from it the ``bid`` price
        (industry de-facto standard with IB according to the literature
        scattered over the Internet)

        Even if set to ``True``, if the data is resampled/kept to a
        timeframe/compression below Seconds/5, no real time bars will be used,
        because IB doesn't serve them below that level

      - ``qcheck`` (default: ``0.5``)

        Time in seconds to wake up if no data is received to give a chance to
        resample/replay packets properly and pass notifications up the chain

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

      - ``latethrough`` (default: ``False``)

        If the data source is resampled/replayed, some ticks may come in too
        late for the already delivered resampled/replayed bar. If this is
        ``True`` those ticks will bet let through in any case.

        Check the Resampler documentation to see who to take those ticks into
        account.

        This can happen especially if ``timeoffset`` is set to ``False``  in
        the ``IBStore`` instance and the TWS server time is not in sync with
        that of the local computer

      - ``tradename`` (default: ``None``)
        Useful for some specific cases like ``CFD`` in which prices are offered
        by one asset and trading happens in a different onel

        - SPY-STK-SMART-USD -> SP500 ETF (will be specified as ``dataname``)

        - SPY-CFD-SMART-USD -> which is the corresponding CFD which offers not
          price tracking but in this case will be the trading asset (specified
          as ``tradename``)

    The default values in the params are the to allow things like ```TICKER``,
    to which the parameter ``sectype`` (default: ``STK``) and ``exchange``
    (default: ``SMART``) are applied.

    Some assets like ``AAPL`` need full specification including ``currency``
    (default: '') whereas others like ``TWTR`` can be simply passed as it is.

      - ``AAPL-STK-SMART-USD`` would be the full specification for dataname

        Or else: ``IBData`` as ``IBData(dataname='AAPL', currency='USD')``
        which uses the default values (``STK`` and ``SMART``) and overrides
        the currency to be ``USD``
    """
    params = (('sectype', 'STK'), ('exchange', 'SMART'), ('currency', ''), ('rtbar', False), ('historical', False), ('what', None), ('useRTH', False), ('qcheck', 0.5), ('backfill_start', True), ('backfill', True), ('backfill_from', None), ('latethrough', False), ('tradename', None))
    _store = ibstore.IBStore
    RTBAR_MINSIZE = (TimeFrame.Seconds, 5)
    (_ST_FROM, _ST_START, _ST_LIVE, _ST_HISTORBACK, _ST_OVER) = range(5)

    def _timeoffset(self):
        if False:
            i = 10
            return i + 15
        return self.ib.timeoffset()

    def _gettz(self):
        if False:
            i = 10
            return i + 15
        tzstr = isinstance(self.p.tz, string_types)
        if self.p.tz is not None and (not tzstr):
            return bt.utils.date.Localizer(self.p.tz)
        if self.contractdetails is None:
            return None
        try:
            import pytz
        except ImportError:
            return None
        tzs = self.p.tz if tzstr else self.contractdetails.m_timeZoneId
        if tzs == 'CST':
            tzs = 'CST6CDT'
        try:
            tz = pytz.timezone(tzs)
        except pytz.UnknownTimeZoneError:
            return None
        return tz

    def islive(self):
        if False:
            print('Hello World!')
        'Returns ``True`` to notify ``Cerebro`` that preloading and runonce\n        should be deactivated'
        return not self.p.historical

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.ib = self._store(**kwargs)
        self.precontract = self.parsecontract(self.p.dataname)
        self.pretradecontract = self.parsecontract(self.p.tradename)

    def setenvironment(self, env):
        if False:
            print('Hello World!')
        'Receives an environment (cerebro) and passes it over to the store it\n        belongs to'
        super(IBData, self).setenvironment(env)
        env.addstore(self.ib)

    def parsecontract(self, dataname):
        if False:
            while True:
                i = 10
        'Parses dataname generates a default contract'
        if dataname is None:
            return None
        exch = self.p.exchange
        curr = self.p.currency
        expiry = ''
        strike = 0.0
        right = ''
        mult = ''
        tokens = iter(dataname.split('-'))
        symbol = next(tokens)
        try:
            sectype = next(tokens)
        except StopIteration:
            sectype = self.p.sectype
        if sectype.isdigit():
            expiry = sectype
            if len(sectype) == 6:
                sectype = 'FUT'
            else:
                sectype = 'OPT'
        if sectype == 'CASH':
            (symbol, curr) = symbol.split('.')
        try:
            exch = next(tokens)
            curr = next(tokens)
            if sectype == 'FUT':
                if not expiry:
                    expiry = next(tokens)
                mult = next(tokens)
                right = next(tokens)
                sectype = 'FOP'
                (strike, mult) = (float(mult), '')
                mult = next(tokens)
            elif sectype == 'OPT':
                if not expiry:
                    expiry = next(tokens)
                strike = float(next(tokens))
                right = next(tokens)
                mult = next(tokens)
        except StopIteration:
            pass
        precon = self.ib.makecontract(symbol=symbol, sectype=sectype, exch=exch, curr=curr, expiry=expiry, strike=strike, right=right, mult=mult)
        return precon

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts the IB connecction and gets the real contract and\n        contractdetails if it exists'
        super(IBData, self).start()
        self.qlive = self.ib.start(data=self)
        self.qhist = None
        self._usertvol = not self.p.rtbar
        tfcomp = (self._timeframe, self._compression)
        if tfcomp < self.RTBAR_MINSIZE:
            self._usertvol = True
        self.contract = None
        self.contractdetails = None
        self.tradecontract = None
        self.tradecontractdetails = None
        if self.p.backfill_from is not None:
            self._state = self._ST_FROM
            self.p.backfill_from.setenvironment(self._env)
            self.p.backfill_from._start()
        else:
            self._state = self._ST_START
        self._statelivereconn = False
        self._subcription_valid = False
        self._storedmsg = dict()
        if not self.ib.connected():
            return
        self.put_notification(self.CONNECTED)
        cds = self.ib.getContractDetails(self.precontract, maxcount=1)
        if cds is not None:
            cdetails = cds[0]
            self.contract = cdetails.contractDetails.m_summary
            self.contractdetails = cdetails.contractDetails
        else:
            self.put_notification(self.DISCONNECTED)
            return
        if self.pretradecontract is None:
            self.tradecontract = self.contract
            self.tradecontractdetails = self.contractdetails
        else:
            cds = self.ib.getContractDetails(self.pretradecontract, maxcount=1)
            if cds is not None:
                cdetails = cds[0]
                self.tradecontract = cdetails.contractDetails.m_summary
                self.tradecontractdetails = cdetails.contractDetails
            else:
                self.put_notification(self.DISCONNECTED)
                return
        if self._state == self._ST_START:
            self._start_finish()
            self._st_start()

    def stop(self):
        if False:
            while True:
                i = 10
        'Stops and tells the store to stop'
        super(IBData, self).stop()
        self.ib.stop()

    def reqdata(self):
        if False:
            print('Hello World!')
        'request real-time data. checks cash vs non-cash) and param useRT'
        if self.contract is None or self._subcription_valid:
            return
        if self._usertvol:
            self.qlive = self.ib.reqMktData(self.contract, self.p.what)
        else:
            self.qlive = self.ib.reqRealTimeBars(self.contract)
        self._subcription_valid = True
        return self.qlive

    def canceldata(self):
        if False:
            return 10
        'Cancels Market Data subscription, checking asset type and rtbar'
        if self.contract is None:
            return
        if self._usertvol:
            self.ib.cancelMktData(self.qlive)
        else:
            self.ib.cancelRealTimeBars(self.qlive)

    def haslivedata(self):
        if False:
            return 10
        return bool(self._storedmsg or self.qlive)

    def _load(self):
        if False:
            print('Hello World!')
        if self.contract is None or self._state == self._ST_OVER:
            return False
        while True:
            if self._state == self._ST_LIVE:
                try:
                    msg = self._storedmsg.pop(None, None) or self.qlive.get(timeout=self._qcheck)
                except queue.Empty:
                    if True:
                        return None
                    if not self._statelivereconn:
                        return None
                    dtend = self.num2date(date2num(datetime.datetime.utcnow()))
                    dtbegin = None
                    if len(self) > 1:
                        dtbegin = self.num2date(self.datetime[-1])
                    self.qhist = self.ib.reqHistoricalDataEx(contract=self.contract, enddate=dtend, begindate=dtbegin, timeframe=self._timeframe, compression=self._compression, what=self.p.what, useRTH=self.p.useRTH, tz=self._tz, sessionend=self.p.sessionend)
                    if self._laststatus != self.DELAYED:
                        self.put_notification(self.DELAYED)
                    self._state = self._ST_HISTORBACK
                    self._statelivereconn = False
                    continue
                if msg is None:
                    self._subcription_valid = False
                    self.put_notification(self.CONNBROKEN)
                    if not self.ib.reconnect(resub=True):
                        self.put_notification(self.DISCONNECTED)
                        return False
                    self._statelivereconn = self.p.backfill
                    continue
                if msg == -354:
                    self.put_notification(self.NOTSUBSCRIBED)
                    return False
                elif msg == -1100:
                    self._subcription_valid = False
                    self._statelivereconn = self.p.backfill
                    continue
                elif msg == -1102:
                    if not self._statelivereconn:
                        self._statelivereconn = self.p.backfill
                    continue
                elif msg == -1101:
                    self._subcription_valid = False
                    if not self._statelivereconn:
                        self._statelivereconn = self.p.backfill
                        self.reqdata()
                    continue
                elif msg == -10225:
                    self._subcription_valid = False
                    if not self._statelivereconn:
                        self._statelivereconn = self.p.backfill
                        self.reqdata()
                    continue
                elif isinstance(msg, integer_types):
                    self.put_notification(self.UNKNOWN, msg)
                    continue
                if not self._statelivereconn:
                    if self._laststatus != self.LIVE:
                        if self.qlive.qsize() <= 1:
                            self.put_notification(self.LIVE)
                    if self._usertvol:
                        ret = self._load_rtvolume(msg)
                    else:
                        ret = self._load_rtbar(msg)
                    if ret:
                        return True
                    continue
                self._storedmsg[None] = msg
                if self._laststatus != self.DELAYED:
                    self.put_notification(self.DELAYED)
                dtend = None
                if len(self) > 1:
                    dtbegin = num2date(self.datetime[-1])
                elif self.fromdate > float('-inf'):
                    dtbegin = num2date(self.fromdate)
                else:
                    dtbegin = None
                dtend = msg.datetime if self._usertvol else msg.time
                self.qhist = self.ib.reqHistoricalDataEx(contract=self.contract, enddate=dtend, begindate=dtbegin, timeframe=self._timeframe, compression=self._compression, what=self.p.what, useRTH=self.p.useRTH, tz=self._tz, sessionend=self.p.sessionend)
                self._state = self._ST_HISTORBACK
                self._statelivereconn = False
                continue
            elif self._state == self._ST_HISTORBACK:
                msg = self.qhist.get()
                if msg is None:
                    self._subcription_valid = False
                    self.put_notification(self.DISCONNECTED)
                    return False
                elif msg == -354:
                    self._subcription_valid = False
                    self.put_notification(self.NOTSUBSCRIBED)
                    return False
                elif msg == -420:
                    self._subcription_valid = False
                    self.put_notification(self.NOTSUBSCRIBED)
                    return False
                elif isinstance(msg, integer_types):
                    self.put_notification(self.UNKNOWN, msg)
                    continue
                if msg.date is not None:
                    if self._load_rtbar(msg, hist=True):
                        return True
                    continue
                if self.p.historical:
                    self.put_notification(self.DISCONNECTED)
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
                if not self._st_start():
                    return False

    def _st_start(self):
        if False:
            for i in range(10):
                print('nop')
        if self.p.historical:
            self.put_notification(self.DELAYED)
            dtend = None
            if self.todate < float('inf'):
                dtend = num2date(self.todate)
            dtbegin = None
            if self.fromdate > float('-inf'):
                dtbegin = num2date(self.fromdate)
            self.qhist = self.ib.reqHistoricalDataEx(contract=self.contract, enddate=dtend, begindate=dtbegin, timeframe=self._timeframe, compression=self._compression, what=self.p.what, useRTH=self.p.useRTH, tz=self._tz, sessionend=self.p.sessionend)
            self._state = self._ST_HISTORBACK
            return True
        if not self.ib.reconnect(resub=True):
            self.put_notification(self.DISCONNECTED)
            self._state = self._ST_OVER
            return False
        self._statelivereconn = self.p.backfill_start
        if self.p.backfill_start:
            self.put_notification(self.DELAYED)
        self._state = self._ST_LIVE
        return True

    def _load_rtbar(self, rtbar, hist=False):
        if False:
            while True:
                i = 10
        dt = date2num(rtbar.time if not hist else rtbar.date)
        if dt < self.lines.datetime[-1] and (not self.p.latethrough):
            return False
        self.lines.datetime[0] = dt
        self.lines.open[0] = rtbar.open
        self.lines.high[0] = rtbar.high
        self.lines.low[0] = rtbar.low
        self.lines.close[0] = rtbar.close
        self.lines.volume[0] = rtbar.volume
        self.lines.openinterest[0] = 0
        return True

    def _load_rtvolume(self, rtvol):
        if False:
            return 10
        dt = date2num(rtvol.datetime)
        if dt < self.lines.datetime[-1] and (not self.p.latethrough):
            return False
        self.lines.datetime[0] = dt
        tick = rtvol.price
        self.lines.open[0] = tick
        self.lines.high[0] = tick
        self.lines.low[0] = tick
        self.lines.close[0] = tick
        self.lines.volume[0] = rtvol.size
        self.lines.openinterest[0] = 0
        return True
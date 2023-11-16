from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, timedelta, tzinfo
import backtrader as bt
from backtrader import TimeFrame, date2num, num2date
from backtrader.feed import DataBase
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import integer_types, queue, string_types, with_metaclass
from backtrader.stores import vcstore

class MetaVCData(DataBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        'Class has already been created ... register'
        super(MetaVCData, cls).__init__(name, bases, dct)
        vcstore.VCStore.DataCls = cls

class VCData(with_metaclass(MetaVCData, DataBase)):
    """VisualChart Data Feed.

    Params:

      - ``qcheck`` (default: ``0.5``)
        Default timeout for waking up to let a resampler/replayer that the
        current bar can be check for due delivery

        The value is only used if a resampling/replaying filter has been
        inserted in the data

      - ``historical`` (default: ``False``)
        If no ``todate`` parameter is supplied (defined in the base class),
        this will force a historical only download if set to ``True``

        If ``todate`` is supplied the same effect is achieved

      - ``milliseconds`` (default: ``True``)
        The bars constructed by *Visual Chart* have this aspect:
        HH:MM:59.999000

        If this parameter is ``True`` a millisecond will be added to this time
        to make it look like: HH::MM + 1:00.000000

      - ``tradename`` (default: ``None``)
        Continous futures cannot be traded but are ideal for data tracking. If
        this parameter is supplied it will be the name of the current future
        which will be the trading asset. Example:

        - 001ES -> ES-Mini continuous supplied as ``dataname``

        - ESU16 -> ES-Mini 2016-09. If this is supplied in ``tradename`` it
          will be the trading asset.

      - ``usetimezones`` (default: ``True``)
        For most markets the time offset information provided by *Visual Chart*
        allows for datetime to be converted to market time (*backtrader* choice
        for representation)

        Some markets are special (``096``) and need special internal coverage
        and timezone support to display in the user expected market time.

        If this parameter is set to ``True`` importing ``pytz`` will be
        attempted to use timezones (default)

        Disabling it will remove timezone usage (may help if the load is
        excesive)
    """
    params = (('qcheck', 0.5), ('historical', False), ('millisecond', True), ('tradename', None), ('usetimezones', True))
    _TOFFSET = timedelta()
    (_ST_START, _ST_FEEDING, _ST_NOTFOUND) = range(3)
    NULLDATE = datetime(1899, 12, 30, 0, 0, 0)
    MILLISECOND = timedelta(microseconds=1000)
    PING_TIMEOUT = 25.0
    _TZS = {'Europe/London': ('011', '024', '027', '036', '049', '092', '114', '033', '034', '035', '043', '054', '096', '300'), 'Europe/Berlin': ('005', '006', '008', '012', '013', '014', '015', '017', '019', '025', '029', '030', '037', '038', '052', '053', '060', '061', '072', '073', '074', '075', '080', '093', '094', '097', '111', '112', '113'), 'Asia/Tokyo': ('031',), 'Australia/Melbourne': ('032',), 'America/Argentina/Buenos_Aires': ('044',), 'America/Sao_Paulo': ('045',), 'America/Mexico_City': ('046',), 'America/Santiago': ('047',), 'US/Eastern': ('003', '004', '009', '010', '028', '040', '041', '055', '090', '095', '099'), 'US/Central': ('001', '002', '020', '021', '022', '023', '056')}
    _TZOUT = {'096.FTSE': 'Europe/London', '096.FTEU3': 'Europe/London', '096.MIB30': 'Europe/Berlin', '096.SSMI': 'Europe/Berlin', '096.HSI': 'Asia/Hong_Kong', '096.BVSP': 'America/Sao_Paulo', '096.MERVAL': 'America/Argentina/Buenos_Aires', '096.DJI': 'US/Eastern', '096.IXIC': 'US/Eastern', '096.NDX': 'US/Eastern'}
    _EXTRA_TIMEOFFSET = ('096',)
    _TIMEFRAME_BACKFILL = {TimeFrame.Ticks: timedelta(days=1), TimeFrame.MicroSeconds: timedelta(days=1), TimeFrame.Seconds: timedelta(days=1), TimeFrame.Minutes: timedelta(days=2), TimeFrame.Days: timedelta(days=365), TimeFrame.Weeks: timedelta(days=365 * 2), TimeFrame.Months: timedelta(days=365 * 5), TimeFrame.Years: timedelta(days=365 * 20)}

    def _timeoffset(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the calculated time offset local equipment -> data server'
        return self._TOFFSET

    def _gettzinput(self):
        if False:
            return 10
        'Returns the timezone to consider for the input data'
        return self._gettz(tzin=True)

    def _gettz(self, tzin=False):
        if False:
            while True:
                i = 10
        'Returns the default output timezone for the data\n\n        This defaults to be the timezone in which the market is traded\n        '
        ptz = self.p.tz
        tzstr = isinstance(ptz, string_types)
        if ptz is not None and (not tzstr):
            return bt.utils.date.Localizer(ptz)
        if self._state == self._ST_NOTFOUND:
            return None
        if not self.p.usetimezones:
            return None
        try:
            import pytz
        except ImportError:
            return None
        if tzstr:
            tzs = ptz
        else:
            tzs = None
            if not tzin:
                if self.p.dataname in self._TZOUT:
                    tzs = self._TZOUT[self.p.dataname]
            if tzs is None:
                for (mktz, mktcodes) in self._TZS.items():
                    if self._mktcode in mktcodes:
                        tzs = mktz
                        break
            if tzs is None:
                return None
            if isinstance(tzs, tzinfo):
                return bt.utils.date.Localizer(tzs)
        if tzs:
            try:
                tz = pytz.timezone(tzs)
            except pytz.UnknownTimeZoneError:
                return None
        else:
            return None
        return tz

    def islive(self):
        if False:
            while True:
                i = 10
        'Returns ``True`` to notify ``Cerebro`` that preloading and runonce\n        should be deactivated'
        return True

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.store = vcstore.VCStore(**kwargs)
        dataname = self.p.dataname
        if dataname[3].isspace():
            dataname = dataname[0:2] + dataname[4:]
            self.p.dataname = dataname
        self._dataname = '010' + self.p.dataname
        self._mktcode = self.p.dataname[0:3]
        self._tradename = tradename = self.p.tradename or self._dataname
        if tradename[3].isspace():
            tradename = tradename[0:2] + tradename[4:]
            self._tradename = tradename

    def setenvironment(self, env):
        if False:
            print('Hello World!')
        'Receives an environment (cerebro) and passes it over to the store it\n        belongs to'
        super(VCData, self).setenvironment(env)
        env.addstore(self.store)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts the VC connecction and gets the real contract and\n        contractdetails if it exists'
        super(VCData, self).start()
        self._state = self._ST_START
        self._newticks = True
        self._pingtmout = self.PING_TIMEOUT
        self.idx = 1
        self.q = None
        self._mktoffset = None
        self._mktoff1 = None
        self._mktoffdiff = None
        if not self.store.connected():
            self.put_notification(self.DISCONNECTED)
            self._state = self._ST_NOTFOUND
            return
        self.put_notification(self.CONNECTED)
        self.qrt = queue.Queue()
        self.store._rtdata(self, self._dataname)
        symfound = self.qrt.get()
        if not symfound:
            self.put_notification(self.NOTSUBSCRIBED)
            self.put_notification(self.DISCONNECTED)
            self._state = self._ST_NOTFOUND
            return
        if self.replaying:
            (self._tf, self._comp) = (self.p.timeframe, self.p.compression)
        else:
            (self._tf, self._comp) = (self._timeframe, self._compression)
        self._ticking = self.store._ticking(self._tf)
        self._syminfo = syminfo = self.store._symboldata(self._dataname)
        self._mktoffset = timedelta(seconds=syminfo.TimeOffset)
        if self.p.millisecond and (not self._ticking):
            self._mktoffset -= self.MILLISECOND
        self._mktoff1 = self._mktoffset
        if self._mktcode in self._EXTRA_TIMEOFFSET:
            self._mktoffset -= timedelta(seconds=3600)
        self._mktoffdiff = self._mktoffset - self._mktoff1
        if self._state == self._ST_START:
            self.put_notification(self.DELAYED)
            self.q = self.store._directdata(self, self._dataname, self._tf, self._comp, self.p.fromdate, self.p.todate, self.p.historical)
            self._state = self._ST_FEEDING

    def stop(self):
        if False:
            return 10
        'Stops and tells the store to stop'
        super(VCData, self).stop()
        if self.q:
            self.store._canceldirectdata(self.q)

    def _setserie(self, serie):
        if False:
            for i in range(10):
                print('nop')
        self._serie = serie

    def haslivedata(self):
        if False:
            for i in range(10):
                print('nop')
        return self._laststatus == self.LIVE and self.q

    def _load(self):
        if False:
            for i in range(10):
                print('nop')
        if self._state == self._ST_NOTFOUND:
            return False
        while True:
            try:
                tmout = self._qcheck * bool(self.resampling)
                msg = self.q.get(timeout=tmout)
            except queue.Empty:
                return None
            if msg is None:
                return False
            if msg == self.store._RT_SHUTDOWN:
                self.put_notification(self.DISCONNECTED)
                return False
            if msg == self.store._RT_DISCONNECTED:
                self.put_notification(self.CONNBROKEN)
                continue
            if msg == self.store._RT_CONNECTED:
                self.put_notification(self.CONNECTED)
                self.put_notification(self.DELAYED)
                continue
            if msg == self.store._RT_LIVE:
                if self._laststatus != self.LIVE:
                    self.put_notification(self.LIVE)
                continue
            if msg == self.store._RT_DELAYED:
                if self._laststatus != self.DELAYED:
                    self.put_notification(self.DELAYED)
                continue
            if isinstance(msg, integer_types):
                self.put_notification(self.UNKNOWN, msg)
                continue
            bar = msg
            self.lines.open[0] = bar.Open
            self.lines.high[0] = bar.High
            self.lines.low[0] = bar.Low
            self.lines.close[0] = bar.Close
            self.lines.volume[0] = bar.Volume
            self.lines.openinterest[0] = bar.OpenInterest
            dt = self.NULLDATE + timedelta(days=bar.Date) - self._mktoffset
            self.lines.datetime[0] = date2num(dt)
            return True

    def _getpingtmout(self):
        if False:
            while True:
                i = 10
        'Returns the actual ping timeout for PumpEvents to wake up and call\n        ping, which will check if the not yet delivered bar can be\n        delivered. The bar may be stalled because vc awaits a new tick and\n        during low negotiation hour this can take several seconds after the\n        actual expected delivery time'
        if self._ticking:
            return -1
        return self._pingtmout

    def OnNewDataSerieBar(self, DataSerie, forcepush=False):
        if False:
            print('Hello World!')
        ssize = DataSerie.Size
        if ssize - self.idx > 1:
            if self._laststatus != self.DELAYED:
                self.q.put(self.store._RT_DELAYED)
        ssize += forcepush or self._ticking
        for idx in range(self.idx, ssize):
            bar = DataSerie.GetBarValues(idx)
            self.q.put(bar)
        if not forcepush and (not self._ticking) and ssize:
            dtnow = datetime.now() - self._TOFFSET
            bar = DataSerie.GetBarValues(ssize)
            dt = self.NULLDATE + timedelta(days=bar.Date) - self._mktoffdiff
            if dtnow < dt:
                if self._laststatus != self.LIVE:
                    self.q.put(self.store._RT_LIVE)
                self._pingtmout = (dt - dtnow).total_seconds() + 0.5
            else:
                self._pingtmout = self.PING_TIMEOUT
                self.q.put(bar)
                ssize += 1
        self.idx = max(1, ssize)

    def ping(self):
        if False:
            for i in range(10):
                print('nop')
        ssize = self._serie.Size
        if self.idx > ssize:
            return
        if self._laststatus == self.CONNBROKEN:
            self._pingtmout = self.PING_TIMEOUT
            return
        dtnow = datetime.now() - self._TOFFSET
        for idx in range(self.idx, ssize + 1):
            bar = self._serie.GetBarValues(self.idx)
            dt = self.NULLDATE + timedelta(days=bar.Date) - self._mktoffdiff
            if dtnow < dt:
                self._pingtmout = (dt - dtnow).total_seconds() + 0.5
                break
            self._pingtmout = self.PING_TIMEOUT
            self.q.put(bar)
            self.idx += 1
    if False:

        def OnInternalEvent(self, p1, p2, p3):
            if False:
                print('Hello World!')
            if p1 != 1:
                return
            if p2 == self.lastconn:
                return
            self.lastconn = p2
            self.store._vcrt_connection(self.store._RT_BASEMSG - p2)

    def OnNewTicks(self, ArrayTicks):
        if False:
            return 10
        aticks = ArrayTicks[0]
        ticks = dict()
        for tick in aticks:
            ticks[tick.Field] = tick
        if self.store.vcrtmod.Field_Description in ticks:
            if self._newticks:
                self._newticks = False
                hasdate = bool(ticks.get(self.store.vcrtmod.Field_Date, False))
                self.qrt.put(hasdate)
                return
        else:
            try:
                tick = ticks[self.store.vcrtmod.Field_Time]
            except KeyError:
                return
            if tick.TickIndex == 0 and self._mktoff1 is not None:
                dttick = self.NULLDATE + timedelta(days=tick.Date) + self._mktoff1
                self._TOFFSET = datetime.now() - dttick
                if self._mktcode in self._EXTRA_TIMEOFFSET:
                    self._TOFFSET -= timedelta(seconds=3600)
                self._vcrt.CancelSymbolFeed(self._dataname, False)

    def debug_ticks(self, ticks):
        if False:
            i = 10
            return i + 15
        print('*' * 50, 'DEBUG OnNewTicks')
        for tick in ticks:
            print('-' * 40)
            print('tick.SymbolCode', tick.SymbolCode.encode('ascii', 'ignore'))
            fname = self.store.vcrtfields.get(tick.Field, tick.Field)
            print('  tick.Field   : {} ({})'.format(fname, tick.Field))
            print('  tick.FieldEx :', tick.FieldEx)
            tdate = tick.Date
            if tdate:
                tdate = self.NULLDATE + timedelta(days=tick.Date)
            print('  tick.Date    :', tdate)
            print('  tick.Index   :', tick.TickIndex)
            print('  tick.Value   :', tick.Value)
            print('  tick.Text    :', tick.Text.encode('ascii', 'ignore'))
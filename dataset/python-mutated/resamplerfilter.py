from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, date, timedelta
from .dataseries import TimeFrame, _Bar
from .utils.py3 import with_metaclass
from . import metabase
from .utils.date import date2num, num2date

class DTFaker(object):

    def __init__(self, data, forcedata=None):
        if False:
            i = 10
            return i + 15
        self.data = data
        self.datetime = self
        self.p = self
        if forcedata is None:
            _dtime = datetime.utcnow() + data._timeoffset()
            self._dt = dt = date2num(_dtime)
            self._dtime = data.num2date(dt)
        else:
            self._dt = forcedata.datetime[0]
            self._dtime = forcedata.datetime.datetime()
        self.sessionend = data.p.sessionend

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.data)

    def __call__(self, idx=0):
        if False:
            print('Hello World!')
        return self._dtime

    def datetime(self, idx=0):
        if False:
            print('Hello World!')
        return self._dtime

    def date(self, idx=0):
        if False:
            while True:
                i = 10
        return self._dtime.date()

    def time(self, idx=0):
        if False:
            print('Hello World!')
        return self._dtime.time()

    @property
    def _calendar(self):
        if False:
            print('Hello World!')
        return self.data._calendar

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        return self._dt if idx == 0 else float('-inf')

    def num2date(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.data.num2date(*args, **kwargs)

    def date2num(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.data.date2num(*args, **kwargs)

    def _getnexteos(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data._getnexteos()

class _BaseResampler(with_metaclass(metabase.MetaParams, object)):
    params = (('bar2edge', True), ('adjbartime', True), ('rightedge', True), ('boundoff', 0), ('timeframe', TimeFrame.Days), ('compression', 1), ('takelate', True), ('sessionend', True))

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.subdays = TimeFrame.Ticks < self.p.timeframe < TimeFrame.Days
        self.subweeks = self.p.timeframe < TimeFrame.Weeks
        self.componly = not self.subdays and data._timeframe == self.p.timeframe and (not self.p.compression % data._compression)
        self.bar = _Bar(maxdate=True)
        self.compcount = 0
        self._firstbar = True
        self.doadjusttime = self.p.bar2edge and self.p.adjbartime and self.subweeks
        self._nexteos = None
        data.resampling = 1
        data.replaying = self.replaying
        data._timeframe = self.p.timeframe
        data._compression = self.p.compression
        self.data = data

    def _latedata(self, data):
        if False:
            print('Hello World!')
        if not self.subdays:
            return False
        return len(data) > 1 and data.datetime[0] <= data.datetime[-1]

    def _checkbarover(self, data, fromcheck=False, forcedata=None):
        if False:
            i = 10
            return i + 15
        chkdata = DTFaker(data, forcedata) if fromcheck else data
        isover = False
        if not self.componly and (not self._barover(chkdata)):
            return isover
        if self.subdays and self.p.bar2edge:
            isover = True
        elif not fromcheck:
            self.compcount += 1
            if not self.compcount % self.p.compression:
                isover = True
        return isover

    def _barover(self, data):
        if False:
            i = 10
            return i + 15
        tframe = self.p.timeframe
        if tframe == TimeFrame.Ticks:
            return self.bar.isopen()
        elif tframe < TimeFrame.Days:
            return self._barover_subdays(data)
        elif tframe == TimeFrame.Days:
            return self._barover_days(data)
        elif tframe == TimeFrame.Weeks:
            return self._barover_weeks(data)
        elif tframe == TimeFrame.Months:
            return self._barover_months(data)
        elif tframe == TimeFrame.Years:
            return self._barover_years(data)

    def _eosset(self):
        if False:
            i = 10
            return i + 15
        if self._nexteos is None:
            (self._nexteos, self._nextdteos) = self.data._getnexteos()
            return

    def _eoscheck(self, data, seteos=True, exact=False):
        if False:
            i = 10
            return i + 15
        if seteos:
            self._eosset()
        equal = data.datetime[0] == self._nextdteos
        grter = data.datetime[0] > self._nextdteos
        if exact:
            ret = equal
        elif grter:
            ret = self.bar.isopen() and self.bar.datetime <= self._nextdteos
        else:
            ret = equal
        if ret:
            self._lasteos = self._nexteos
            self._lastdteos = self._nextdteos
            self._nexteos = None
            self._nextdteos = float('-inf')
        return ret

    def _barover_days(self, data):
        if False:
            return 10
        return self._eoscheck(data)

    def _barover_weeks(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.data._calendar is None:
            (year, week, _) = data.num2date(self.bar.datetime).date().isocalendar()
            yearweek = year * 100 + week
            (baryear, barweek, _) = data.datetime.date().isocalendar()
            bar_yearweek = baryear * 100 + barweek
            return bar_yearweek > yearweek
        else:
            return data._calendar.last_weekday(data.datetime.date())

    def _barover_months(self, data):
        if False:
            i = 10
            return i + 15
        dt = data.num2date(self.bar.datetime).date()
        yearmonth = dt.year * 100 + dt.month
        bardt = data.datetime.datetime()
        bar_yearmonth = bardt.year * 100 + bardt.month
        return bar_yearmonth > yearmonth

    def _barover_years(self, data):
        if False:
            i = 10
            return i + 15
        return data.datetime.datetime().year > data.num2date(self.bar.datetime).year

    def _gettmpoint(self, tm):
        if False:
            return 10
        'Returns the point of time intraday for a given time according to the\n        timeframe\n\n          - Ex 1: 00:05:00 in minutes -> point = 5\n          - Ex 2: 00:05:20 in seconds -> point = 5 * 60 + 20 = 320\n        '
        point = tm.hour * 60 + tm.minute
        restpoint = 0
        if self.p.timeframe < TimeFrame.Minutes:
            point = point * 60 + tm.second
            if self.p.timeframe < TimeFrame.Seconds:
                point = point * 1000000.0 + tm.microsecond
            else:
                restpoint = tm.microsecond
        else:
            restpoint = tm.second + tm.microsecond
        point += self.p.boundoff
        return (point, restpoint)

    def _barover_subdays(self, data):
        if False:
            i = 10
            return i + 15
        if self._eoscheck(data):
            return True
        if data.datetime[0] < self.bar.datetime:
            return False
        tm = num2date(self.bar.datetime).time()
        bartm = num2date(data.datetime[0]).time()
        (point, _) = self._gettmpoint(tm)
        (barpoint, _) = self._gettmpoint(bartm)
        ret = False
        if barpoint > point:
            if not self.p.bar2edge:
                ret = True
            elif self.p.compression == 1:
                ret = True
            else:
                point_comp = point // self.p.compression
                barpoint_comp = barpoint // self.p.compression
                if barpoint_comp > point_comp:
                    ret = True
        return ret

    def check(self, data, _forcedata=None):
        if False:
            print('Hello World!')
        'Called to check if the current stored bar has to be delivered in\n        spite of the data not having moved forward. If no ticks from a live\n        feed come in, a 5 second resampled bar could be delivered 20 seconds\n        later. When this method is called the wall clock (incl data time\n        offset) is called to check if the time has gone so far as to have to\n        deliver the already stored data\n        '
        if not self.bar.isopen():
            return
        return self(data, fromcheck=True, forcedata=_forcedata)

    def _dataonedge(self, data):
        if False:
            print('Hello World!')
        if not self.subweeks:
            if data._calendar is None:
                return (False, True)
            tframe = self.p.timeframe
            ret = False
            if tframe == TimeFrame.Weeks:
                ret = data._calendar.last_weekday(data.datetime.date())
            elif tframe == TimeFrame.Months:
                ret = data._calendar.last_monthday(data.datetime.date())
            elif tframe == TimeFrame.Years:
                ret = data._calendar.last_yearday(data.datetime.date())
            if ret:
                docheckover = False
                self.compcount += 1
                ret = not self.compcount % self.p.compression
            else:
                docheckover = True
            return (ret, docheckover)
        if self._eoscheck(data, exact=True):
            return (True, True)
        if self.subdays:
            (point, prest) = self._gettmpoint(data.datetime.time())
            if prest:
                return (False, True)
            (bound, brest) = divmod(point, self.p.compression)
            return (brest == 0 and point == bound * self.p.compression, True)
        if False and self.p.sessionend:
            bdtime = data.datetime.datetime()
            bsend = datetime.combine(bdtime.date(), data.p.sessionend)
            return bdtime == bsend
        return (False, True)

    def _calcadjtime(self, greater=False):
        if False:
            return 10
        if self._nexteos is None:
            return self._lastdteos
        dt = self.data.num2date(self.bar.datetime)
        tm = dt.time()
        (point, _) = self._gettmpoint(tm)
        point = point // self.p.compression
        point += self.p.rightedge
        point *= self.p.compression
        extradays = 0
        if self.p.timeframe == TimeFrame.Minutes:
            (ph, pm) = divmod(point, 60)
            ps = 0
            pus = 0
        elif self.p.timeframe == TimeFrame.Seconds:
            (ph, pm) = divmod(point, 60 * 60)
            (pm, ps) = divmod(pm, 60)
            pus = 0
        elif self.p.timeframe <= TimeFrame.MicroSeconds:
            (ph, pm) = divmod(point, 60 * 60 * 1000000.0)
            (pm, psec) = divmod(pm, 60 * 1000000.0)
            (ps, pus) = divmod(psec, 1000000.0)
        elif self.p.timeframe == TimeFrame.Days:
            eost = self._nexteos.time()
            ph = eost.hour
            pm = eost.minute
            ps = eost.second
            pus = eost.microsecond
        if ph > 23:
            extradays = ph // 24
            ph %= 24
        dt = dt.replace(hour=int(ph), minute=int(pm), second=int(ps), microsecond=int(pus))
        if extradays:
            dt += timedelta(days=extradays)
        dtnum = self.data.date2num(dt)
        return dtnum

    def _adjusttime(self, greater=False, forcedata=None):
        if False:
            print('Hello World!')
        '\n        Adjusts the time of calculated bar (from underlying data source) by\n        using the timeframe to the appropriate boundary, with compression taken\n        into account\n\n        Depending on param ``rightedge`` uses the starting boundary or the\n        ending one\n        '
        dtnum = self._calcadjtime(greater=greater)
        if greater and dtnum <= self.bar.datetime:
            return False
        self.bar.datetime = dtnum
        return True

class Resampler(_BaseResampler):
    """This class resamples data of a given timeframe to a larger timeframe.

    Params

      - bar2edge (default: True)

        resamples using time boundaries as the target. For example with a
        "ticks -> 5 seconds" the resulting 5 seconds bars will be aligned to
        xx:00, xx:05, xx:10 ...

      - adjbartime (default: True)

        Use the time at the boundary to adjust the time of the delivered
        resampled bar instead of the last seen timestamp. If resampling to "5
        seconds" the time of the bar will be adjusted for example to hh:mm:05
        even if the last seen timestamp was hh:mm:04.33

        .. note::

           Time will only be adjusted if "bar2edge" is True. It wouldn't make
           sense to adjust the time if the bar has not been aligned to a
           boundary

      - rightedge (default: True)

        Use the right edge of the time boundaries to set the time.

        If False and compressing to 5 seconds the time of a resampled bar for
        seconds between hh:mm:00 and hh:mm:04 will be hh:mm:00 (the starting
        boundary

        If True the used boundary for the time will be hh:mm:05 (the ending
        boundary)
    """
    params = (('bar2edge', True), ('adjbartime', True), ('rightedge', True))
    replaying = False

    def last(self, data):
        if False:
            i = 10
            return i + 15
        'Called when the data is no longer producing bars\n\n        Can be called multiple times. It has the chance to (for example)\n        produce extra bars which may still be accumulated and have to be\n        delivered\n        '
        if self.bar.isopen():
            if self.doadjusttime:
                self._adjusttime()
            data._add2stack(self.bar.lvalues())
            self.bar.bstart(maxdate=True)
            return True
        return False

    def __call__(self, data, fromcheck=False, forcedata=None):
        if False:
            while True:
                i = 10
        'Called for each set of values produced by the data source'
        consumed = False
        onedge = False
        docheckover = True
        if not fromcheck:
            if self._latedata(data):
                if not self.p.takelate:
                    data.backwards()
                    return True
                self.bar.bupdate(data)
                self.bar.datetime = data.datetime[-1] + 1e-06
                data.backwards()
                return True
            if self.componly:
                (_, self._lastdteos) = self.data._getnexteos()
                consumed = True
            else:
                (onedge, docheckover) = self._dataonedge(data)
                consumed = onedge
        if consumed:
            self.bar.bupdate(data)
            data.backwards()
        cond = self.bar.isopen()
        if cond:
            if not onedge:
                if docheckover:
                    cond = self._checkbarover(data, fromcheck=fromcheck, forcedata=forcedata)
        if cond:
            dodeliver = False
            if forcedata is not None:
                tframe = self.p.timeframe
                if tframe == TimeFrame.Ticks:
                    dodeliver = True
                elif tframe == TimeFrame.Minutes:
                    dtnum = self._calcadjtime(greater=True)
                    dodeliver = dtnum <= forcedata.datetime[0]
                elif tframe == TimeFrame.Days:
                    dtnum = self._calcadjtime(greater=True)
                    dodeliver = dtnum <= forcedata.datetime[0]
            else:
                dodeliver = True
            if dodeliver:
                if not onedge and self.doadjusttime:
                    self._adjusttime(greater=True, forcedata=forcedata)
                data._add2stack(self.bar.lvalues())
                self.bar.bstart(maxdate=True)
        if not fromcheck:
            if not consumed:
                self.bar.bupdate(data)
                data.backwards()
        return True

class Replayer(_BaseResampler):
    """This class replays data of a given timeframe to a larger timeframe.

    It simulates the action of the market by slowly building up (for ex.) a
    daily bar from tick/seconds/minutes data

    Only when the bar is complete will the "length" of the data be changed
    effectively delivering a closed bar

    Params

      - bar2edge (default: True)

        replays using time boundaries as the target of the closed bar. For
        example with a "ticks -> 5 seconds" the resulting 5 seconds bars will
        be aligned to xx:00, xx:05, xx:10 ...

      - adjbartime (default: False)

        Use the time at the boundary to adjust the time of the delivered
        resampled bar instead of the last seen timestamp. If resampling to "5
        seconds" the time of the bar will be adjusted for example to hh:mm:05
        even if the last seen timestamp was hh:mm:04.33

        .. note::

           Time will only be adjusted if "bar2edge" is True. It wouldn't make
           sense to adjust the time if the bar has not been aligned to a
           boundary

        .. note:: if this parameter is True an extra tick with the *adjusted*
                  time will be introduced at the end of the *replayed* bar

      - rightedge (default: True)

        Use the right edge of the time boundaries to set the time.

        If False and compressing to 5 seconds the time of a resampled bar for
        seconds between hh:mm:00 and hh:mm:04 will be hh:mm:00 (the starting
        boundary

        If True the used boundary for the time will be hh:mm:05 (the ending
        boundary)
    """
    params = (('bar2edge', True), ('adjbartime', False), ('rightedge', True))
    replaying = True

    def __call__(self, data, fromcheck=False, forcedata=None):
        if False:
            return 10
        consumed = False
        onedge = False
        takinglate = False
        docheckover = True
        if not fromcheck:
            if self._latedata(data):
                if not self.p.takelate:
                    data.backwards(force=True)
                    return True
                consumed = True
                takinglate = True
            elif self.componly:
                consumed = True
            else:
                (onedge, docheckover) = self._dataonedge(data)
                consumed = onedge
            data._tick_fill(force=True)
        if consumed:
            self.bar.bupdate(data)
            if takinglate:
                self.bar.datetime = data.datetime[-1] + 1e-06
        cond = onedge
        if not cond:
            if docheckover:
                cond = self._checkbarover(data, fromcheck=fromcheck)
        if cond:
            if not onedge and self.doadjusttime:
                adjusted = self._adjusttime(greater=True)
                if adjusted:
                    ago = 0 if consumed or fromcheck else -1
                    data._updatebar(self.bar.lvalues(), forward=False, ago=ago)
                if not fromcheck:
                    if not consumed:
                        self.bar.bupdate(data, reopen=True)
                        data._save2stack(erase=True, force=True)
                    else:
                        self.bar.bstart(maxdate=True)
                        self._firstbar = True
                else:
                    self.bar.bstart(maxdate=True)
                    self._firstbar = True
                    if adjusted:
                        data._save2stack(erase=True, force=True)
            elif not fromcheck:
                if not consumed:
                    self.bar.bupdate(data, reopen=True)
                else:
                    if not self._firstbar:
                        data.backwards(force=True)
                    data._updatebar(self.bar.lvalues(), forward=False, ago=0)
                    self.bar.bstart(maxdate=True)
                    self._firstbar = True
        elif not fromcheck:
            if not consumed:
                self.bar.bupdate(data)
            if not self._firstbar:
                data.backwards(force=True)
            data._updatebar(self.bar.lvalues(), forward=False, ago=0)
            self._firstbar = False
        return False

class ResamplerTicks(Resampler):
    params = (('timeframe', TimeFrame.Ticks),)

class ResamplerSeconds(Resampler):
    params = (('timeframe', TimeFrame.Seconds),)

class ResamplerMinutes(Resampler):
    params = (('timeframe', TimeFrame.Minutes),)

class ResamplerDaily(Resampler):
    params = (('timeframe', TimeFrame.Days),)

class ResamplerWeekly(Resampler):
    params = (('timeframe', TimeFrame.Weeks),)

class ResamplerMonthly(Resampler):
    params = (('timeframe', TimeFrame.Months),)

class ResamplerYearly(Resampler):
    params = (('timeframe', TimeFrame.Years),)

class ReplayerTicks(Replayer):
    params = (('timeframe', TimeFrame.Ticks),)

class ReplayerSeconds(Replayer):
    params = (('timeframe', TimeFrame.Seconds),)

class ReplayerMinutes(Replayer):
    params = (('timeframe', TimeFrame.Minutes),)

class ReplayerDaily(Replayer):
    params = (('timeframe', TimeFrame.Days),)

class ReplayerWeekly(Replayer):
    params = (('timeframe', TimeFrame.Weeks),)

class ReplayerMonthly(Replayer):
    params = (('timeframe', TimeFrame.Months),)
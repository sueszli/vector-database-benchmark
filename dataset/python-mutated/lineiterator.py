from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import operator
import sys
from .utils.py3 import map, range, zip, with_metaclass, string_types
from .utils import DotDict
from .lineroot import LineRoot, LineSingle
from .linebuffer import LineActions, LineNum
from .lineseries import LineSeries, LineSeriesMaker
from .dataseries import DataSeries
from . import metabase

class MetaLineIterator(LineSeries.__class__):

    def donew(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        (_obj, args, kwargs) = super(MetaLineIterator, cls).donew(*args, **kwargs)
        _obj._lineiterators = collections.defaultdict(list)
        mindatas = _obj._mindatas
        lastarg = 0
        _obj.datas = []
        for arg in args:
            if isinstance(arg, LineRoot):
                _obj.datas.append(LineSeriesMaker(arg))
            elif not mindatas:
                break
            else:
                try:
                    _obj.datas.append(LineSeriesMaker(LineNum(arg)))
                except:
                    break
            mindatas = max(0, mindatas - 1)
            lastarg += 1
        newargs = args[lastarg:]
        if not _obj.datas and isinstance(_obj, (IndicatorBase, ObserverBase)):
            _obj.datas = _obj._owner.datas[0:mindatas]
        _obj.ddatas = {x: None for x in _obj.datas}
        if _obj.datas:
            _obj.data = data = _obj.datas[0]
            for (l, line) in enumerate(data.lines):
                linealias = data._getlinealias(l)
                if linealias:
                    setattr(_obj, 'data_%s' % linealias, line)
                setattr(_obj, 'data_%d' % l, line)
            for (d, data) in enumerate(_obj.datas):
                setattr(_obj, 'data%d' % d, data)
                for (l, line) in enumerate(data.lines):
                    linealias = data._getlinealias(l)
                    if linealias:
                        setattr(_obj, 'data%d_%s' % (d, linealias), line)
                    setattr(_obj, 'data%d_%d' % (d, l), line)
        _obj.dnames = DotDict([(d._name, d) for d in _obj.datas if getattr(d, '_name', '')])
        return (_obj, newargs, kwargs)

    def dopreinit(cls, _obj, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (_obj, args, kwargs) = super(MetaLineIterator, cls).dopreinit(_obj, *args, **kwargs)
        _obj.datas = _obj.datas or [_obj._owner]
        _obj._clock = _obj.datas[0]
        _obj._minperiod = max([x._minperiod for x in _obj.datas] or [_obj._minperiod])
        for line in _obj.lines:
            line.addminperiod(_obj._minperiod)
        return (_obj, args, kwargs)

    def dopostinit(cls, _obj, *args, **kwargs):
        if False:
            print('Hello World!')
        (_obj, args, kwargs) = super(MetaLineIterator, cls).dopostinit(_obj, *args, **kwargs)
        _obj._minperiod = max([x._minperiod for x in _obj.lines])
        _obj._periodrecalc()
        if _obj._owner is not None:
            _obj._owner.addindicator(_obj)
        return (_obj, args, kwargs)

class LineIterator(with_metaclass(MetaLineIterator, LineSeries)):
    _nextforce = False
    _mindatas = 1
    _ltype = LineSeries.IndType
    plotinfo = dict(plot=True, subplot=True, plotname='', plotskip=False, plotabove=False, plotlinelabels=False, plotlinevalues=True, plotvaluetags=True, plotymargin=0.0, plotyhlines=[], plotyticks=[], plothlines=[], plotforce=False, plotmaster=None)

    def _periodrecalc(self):
        if False:
            while True:
                i = 10
        indicators = self._lineiterators[LineIterator.IndType]
        indperiods = [ind._minperiod for ind in indicators]
        indminperiod = max(indperiods or [self._minperiod])
        self.updateminperiod(indminperiod)

    def _stage2(self):
        if False:
            i = 10
            return i + 15
        super(LineIterator, self)._stage2()
        for data in self.datas:
            data._stage2()
        for lineiterators in self._lineiterators.values():
            for lineiterator in lineiterators:
                lineiterator._stage2()

    def _stage1(self):
        if False:
            print('Hello World!')
        super(LineIterator, self)._stage1()
        for data in self.datas:
            data._stage1()
        for lineiterators in self._lineiterators.values():
            for lineiterator in lineiterators:
                lineiterator._stage1()

    def getindicators(self):
        if False:
            for i in range(10):
                print('nop')
        return self._lineiterators[LineIterator.IndType]

    def getindicators_lines(self):
        if False:
            for i in range(10):
                print('nop')
        return [x for x in self._lineiterators[LineIterator.IndType] if hasattr(x.lines, 'getlinealiases')]

    def getobservers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._lineiterators[LineIterator.ObsType]

    def addindicator(self, indicator):
        if False:
            return 10
        self._lineiterators[indicator._ltype].append(indicator)
        if getattr(indicator, '_nextforce', False):
            o = self
            while o is not None:
                if o._ltype == LineIterator.StratType:
                    o.cerebro._disable_runonce()
                    break
                o = o._owner

    def bindlines(self, owner=None, own=None):
        if False:
            i = 10
            return i + 15
        if not owner:
            owner = 0
        if isinstance(owner, string_types):
            owner = [owner]
        elif not isinstance(owner, collections.Iterable):
            owner = [owner]
        if not own:
            own = range(len(owner))
        if isinstance(own, string_types):
            own = [own]
        elif not isinstance(own, collections.Iterable):
            own = [own]
        for (lineowner, lineown) in zip(owner, own):
            if isinstance(lineowner, string_types):
                lownerref = getattr(self._owner.lines, lineowner)
            else:
                lownerref = self._owner.lines[lineowner]
            if isinstance(lineown, string_types):
                lownref = getattr(self.lines, lineown)
            else:
                lownref = self.lines[lineown]
            lownref.addbinding(lownerref)
        return self
    bind2lines = bindlines
    bind2line = bind2lines

    def _next(self):
        if False:
            for i in range(10):
                print('nop')
        clock_len = self._clk_update()
        for indicator in self._lineiterators[LineIterator.IndType]:
            indicator._next()
        self._notify()
        if self._ltype == LineIterator.StratType:
            minperstatus = self._getminperstatus()
            if minperstatus < 0:
                self.next()
            elif minperstatus == 0:
                self.nextstart()
            else:
                self.prenext()
        elif clock_len > self._minperiod:
            self.next()
        elif clock_len == self._minperiod:
            self.nextstart()
        elif clock_len:
            self.prenext()

    def _clk_update(self):
        if False:
            i = 10
            return i + 15
        clock_len = len(self._clock)
        if clock_len != len(self):
            self.forward()
        return clock_len

    def _once(self):
        if False:
            for i in range(10):
                print('nop')
        self.forward(size=self._clock.buflen())
        for indicator in self._lineiterators[LineIterator.IndType]:
            indicator._once()
        for observer in self._lineiterators[LineIterator.ObsType]:
            observer.forward(size=self.buflen())
        for data in self.datas:
            data.home()
        for indicator in self._lineiterators[LineIterator.IndType]:
            indicator.home()
        for observer in self._lineiterators[LineIterator.ObsType]:
            observer.home()
        self.home()
        self.preonce(0, self._minperiod - 1)
        self.oncestart(self._minperiod - 1, self._minperiod)
        self.once(self._minperiod, self.buflen())
        for line in self.lines:
            line.oncebinding()

    def preonce(self, start, end):
        if False:
            return 10
        pass

    def oncestart(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        self.once(start, end)

    def once(self, start, end):
        if False:
            print('Hello World!')
        pass

    def prenext(self):
        if False:
            i = 10
            return i + 15
        '\n        This method will be called before the minimum period of all\n        datas/indicators have been meet for the strategy to start executing\n        '
        pass

    def nextstart(self):
        if False:
            print('Hello World!')
        '\n        This method will be called once, exactly when the minimum period for\n        all datas/indicators have been meet. The default behavior is to call\n        next\n        '
        self.next()

    def next(self):
        if False:
            return 10
        '\n        This method will be called for all remaining data points when the\n        minimum period for all datas/indicators have been meet.\n        '
        pass

    def _addnotification(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def _notify(self):
        if False:
            i = 10
            return i + 15
        pass

    def _plotinit(self):
        if False:
            while True:
                i = 10
        pass

    def qbuffer(self, savemem=0):
        if False:
            for i in range(10):
                print('nop')
        if savemem:
            for line in self.lines:
                line.qbuffer()
        for obj in self._lineiterators[self.IndType]:
            obj.qbuffer(savemem=1)
        for data in self.datas:
            data.minbuffer(self._minperiod)

class DataAccessor(LineIterator):
    PriceClose = DataSeries.Close
    PriceLow = DataSeries.Low
    PriceHigh = DataSeries.High
    PriceOpen = DataSeries.Open
    PriceVolume = DataSeries.Volume
    PriceOpenInteres = DataSeries.OpenInterest
    PriceDateTime = DataSeries.DateTime

class IndicatorBase(DataAccessor):
    pass

class ObserverBase(DataAccessor):
    pass

class StrategyBase(DataAccessor):
    pass

class SingleCoupler(LineActions):

    def __init__(self, cdata, clock=None):
        if False:
            print('Hello World!')
        super(SingleCoupler, self).__init__()
        self._clock = clock if clock is not None else self._owner
        self.cdata = cdata
        self.dlen = 0
        self.val = float('NaN')

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.cdata) > self.dlen:
            self.val = self.cdata[0]
            self.dlen += 1
        self[0] = self.val

class MultiCoupler(LineIterator):
    _ltype = LineIterator.IndType

    def __init__(self):
        if False:
            print('Hello World!')
        super(MultiCoupler, self).__init__()
        self.dlen = 0
        self.dsize = self.fullsize()
        self.dvals = [float('NaN')] * self.dsize

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.data) > self.dlen:
            self.dlen += 1
            for i in range(self.dsize):
                self.dvals[i] = self.data.lines[i][0]
        for i in range(self.dsize):
            self.lines[i][0] = self.dvals[i]

def LinesCoupler(cdata, clock=None, **kwargs):
    if False:
        print('Hello World!')
    if isinstance(cdata, LineSingle):
        return SingleCoupler(cdata, clock)
    cdatacls = cdata.__class__
    try:
        LinesCoupler.counter += 1
    except AttributeError:
        LinesCoupler.counter = 0
    nclsname = str('LinesCoupler_%d' % LinesCoupler.counter)
    ncls = type(nclsname, (MultiCoupler,), {})
    thismod = sys.modules[LinesCoupler.__module__]
    setattr(thismod, ncls.__name__, ncls)
    ncls.lines = cdatacls.lines
    ncls.params = cdatacls.params
    ncls.plotinfo = cdatacls.plotinfo
    ncls.plotlines = cdatacls.plotlines
    obj = ncls(cdata, **kwargs)
    if clock is None:
        clock = getattr(cdata, '_clock', None)
        if clock is not None:
            nclock = getattr(clock, '_clock', None)
            if nclock is not None:
                clock = nclock
            else:
                nclock = getattr(clock, 'data', None)
                if nclock is not None:
                    clock = nclock
        if clock is None:
            clock = obj._owner
    obj._clock = clock
    return obj
LineCoupler = LinesCoupler
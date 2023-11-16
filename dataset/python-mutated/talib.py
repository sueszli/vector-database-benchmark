from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import backtrader as bt
from backtrader.utils.py3 import with_metaclass
try:
    import talib
except ImportError:
    __all__ = []
else:
    import numpy as np
    import talib.abstract
    MA_Type = talib.MA_Type
    R_TA_FUNC_FLAGS = dict(zip(talib.abstract.TA_FUNC_FLAGS.values(), talib.abstract.TA_FUNC_FLAGS.keys()))
    FUNC_FLAGS_SAMESCALE = 16777216
    FUNC_FLAGS_UNSTABLE = 134217728
    FUNC_FLAGS_CANDLESTICK = 268435456
    R_TA_OUTPUT_FLAGS = dict(zip(talib.abstract.TA_OUTPUT_FLAGS.values(), talib.abstract.TA_OUTPUT_FLAGS.keys()))
    OUT_FLAGS_LINE = 1
    OUT_FLAGS_DOTTED = 2
    OUT_FLAGS_DASH = 4
    OUT_FLAGS_HISTO = 16
    OUT_FLAGS_UPPER = 2048
    OUT_FLAGS_LOWER = 4096

    class _MetaTALibIndicator(bt.Indicator.__class__):
        _refname = '_taindcol'
        _taindcol = dict()
        _KNOWN_UNSTABLE = ['SAR']

        def dopostinit(cls, _obj, *args, **kwargs):
            if False:
                print('Hello World!')
            res = super(_MetaTALibIndicator, cls).dopostinit(_obj, *args, **kwargs)
            (_obj, args, kwargs) = res
            _obj._tabstract.set_function_args(**_obj.p._getkwargs())
            _obj._lookback = lookback = _obj._tabstract.lookback + 1
            _obj.updateminperiod(lookback)
            if _obj._unstable:
                _obj._lookback = 0
            elif cls.__name__ in cls._KNOWN_UNSTABLE:
                _obj._lookback = 0
            cerebro = bt.metabase.findowner(_obj, bt.Cerebro)
            tafuncinfo = _obj._tabstract.info
            _obj._tafunc = getattr(talib, tafuncinfo['name'], None)
            return (_obj, args, kwargs)

    class _TALibIndicator(with_metaclass(_MetaTALibIndicator, bt.Indicator)):
        CANDLEOVER = 1.02
        CANDLEREF = 1

        @classmethod
        def _subclass(cls, name):
            if False:
                return 10
            clsmodule = sys.modules[cls.__module__]
            _tabstract = talib.abstract.Function(name)
            iscandle = False
            unstable = False
            plotinfo = dict()
            fflags = _tabstract.function_flags or []
            for fflag in fflags:
                rfflag = R_TA_FUNC_FLAGS[fflag]
                if rfflag == FUNC_FLAGS_SAMESCALE:
                    plotinfo['subplot'] = False
                elif rfflag == FUNC_FLAGS_UNSTABLE:
                    unstable = True
                elif rfflag == FUNC_FLAGS_CANDLESTICK:
                    plotinfo['subplot'] = False
                    plotinfo['plotlinelabels'] = True
                    iscandle = True
            lines = _tabstract.output_names
            output_flags = _tabstract.output_flags
            plotlines = dict()
            samecolor = False
            for lname in lines:
                oflags = output_flags.get(lname, None)
                pline = dict()
                for oflag in oflags or []:
                    orflag = R_TA_OUTPUT_FLAGS[oflag]
                    if orflag & OUT_FLAGS_LINE:
                        if not iscandle:
                            pline['ls'] = '-'
                        else:
                            pline['_plotskip'] = True
                    elif orflag & OUT_FLAGS_DASH:
                        pline['ls'] = '--'
                    elif orflag & OUT_FLAGS_DOTTED:
                        pline['ls'] = ':'
                    elif orflag & OUT_FLAGS_HISTO:
                        pline['_method'] = 'bar'
                    if samecolor:
                        pline['_samecolor'] = True
                    if orflag & OUT_FLAGS_LOWER:
                        samecolor = False
                    elif orflag & OUT_FLAGS_UPPER:
                        samecolor = True
                if pline:
                    plotlines[lname] = pline
            if iscandle:
                pline = dict()
                pline['_name'] = name
                lname = '_candleplot'
                lines.append(lname)
                pline['ls'] = ''
                pline['marker'] = 'd'
                pline['markersize'] = '7.0'
                pline['fillstyle'] = 'full'
                plotlines[lname] = pline
            clsdict = {'__module__': cls.__module__, '__doc__': str(_tabstract), '_tabstract': _tabstract, '_iscandle': iscandle, '_unstable': unstable, 'params': _tabstract.get_parameters(), 'lines': tuple(lines), 'plotinfo': plotinfo, 'plotlines': plotlines}
            newcls = type(str(name), (cls,), clsdict)
            setattr(clsmodule, str(name), newcls)

        def oncestart(self, start, end):
            if False:
                while True:
                    i = 10
            pass

        def once(self, start, end):
            if False:
                print('Hello World!')
            import array
            narrays = [np.array(x.lines[0].array) for x in self.datas]
            output = self._tafunc(*narrays, **self.p._getkwargs())
            fsize = self.size()
            lsize = fsize - self._iscandle
            if lsize == 1:
                self.lines[0].array = array.array(str('d'), output)
                if fsize > lsize:
                    candleref = narrays[self.CANDLEREF] * self.CANDLEOVER
                    output2 = candleref * (output / 100.0)
                    self.lines[1].array = array.array(str('d'), output2)
            else:
                for (i, o) in enumerate(output):
                    self.lines[i].array = array.array(str('d'), o)

        def next(self):
            if False:
                print('Hello World!')
            size = self._lookback or len(self)
            narrays = [np.array(x.lines[0].get(size=size)) for x in self.datas]
            out = self._tafunc(*narrays, **self.p._getkwargs())
            fsize = self.size()
            lsize = fsize - self._iscandle
            if lsize == 1:
                self.lines[0][0] = o = out[-1]
                if fsize > lsize:
                    candleref = narrays[self.CANDLEREF][-1] * self.CANDLEOVER
                    o2 = candleref * (o / 100.0)
                    self.lines[1][0] = o2
            else:
                for (i, o) in enumerate(out):
                    self.lines[i][0] = o[-1]
    tafunctions = talib.get_functions()
    for tafunc in tafunctions:
        _TALibIndicator._subclass(tafunc)
    __all__ = tafunctions + ['MA_Type', '_TALibIndicator']
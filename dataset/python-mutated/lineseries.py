"""

.. module:: lineroot

Defines LineSeries and Descriptors inside of it for classes that hold multiple
lines at once.

.. moduleauthor:: Daniel Rodriguez

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from .utils.py3 import map, range, string_types, with_metaclass
from .linebuffer import LineBuffer, LineActions, LinesOperation, LineDelay, NAN
from .lineroot import LineRoot, LineSingle, LineMultiple
from .metabase import AutoInfoClass
from . import metabase

class LineAlias(object):
    """ Descriptor class that store a line reference and returns that line
    from the owner

    Keyword Args:
        line (int): reference to the line that will be returned from
        owner's *lines* buffer

    As a convenience the __set__ method of the descriptor is used not set
    the *line* reference because this is a constant along the live of the
    descriptor instance, but rather to set the value of the *line* at the
    instant '0' (the current one)
    """

    def __init__(self, line):
        if False:
            return 10
        self.line = line

    def __get__(self, obj, cls=None):
        if False:
            while True:
                i = 10
        return obj.lines[self.line]

    def __set__(self, obj, value):
        if False:
            i = 10
            return i + 15
        '\n        A line cannot be "set" once it has been created. But the values\n        inside the line can be "set". This is achieved by adding a binding\n        to the line inside "value"\n        '
        if isinstance(value, LineMultiple):
            value = value.lines[0]
        if not isinstance(value, LineActions):
            value = value(0)
        value.addbinding(obj.lines[self.line])

class Lines(object):
    """
    Defines an "array" of lines which also has most of the interface of
    a LineBuffer class (forward, rewind, advance...).

    This interface operations are passed to the lines held by self

    The class can autosubclass itself (_derive) to hold new lines keeping them
    in the defined order.
    """
    _getlinesbase = classmethod(lambda cls: ())
    _getlines = classmethod(lambda cls: ())
    _getlinesextra = classmethod(lambda cls: 0)
    _getlinesextrabase = classmethod(lambda cls: 0)

    @classmethod
    def _derive(cls, name, lines, extralines, otherbases, linesoverride=False, lalias=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a subclass of this class with the lines of this class as\n        initial input for the subclass. It will include num "extralines" and\n        lines present in "otherbases"\n\n        "name" will be used as the suffix of the final class name\n\n        "linesoverride": if True the lines of all bases will be discarded and\n        the baseclass will be the topmost class "Lines". This is intended to\n        create a new hierarchy\n        '
        obaseslines = ()
        obasesextralines = 0
        for otherbase in otherbases:
            if isinstance(otherbase, tuple):
                obaseslines += otherbase
            else:
                obaseslines += otherbase._getlines()
                obasesextralines += otherbase._getlinesextra()
        if not linesoverride:
            baselines = cls._getlines() + obaseslines
            baseextralines = cls._getlinesextra() + obasesextralines
        else:
            baselines = ()
            baseextralines = 0
        clslines = baselines + lines
        clsextralines = baseextralines + extralines
        lines2add = obaseslines + lines
        basecls = cls if not linesoverride else Lines
        newcls = type(str(cls.__name__ + '_' + name), (basecls,), {})
        clsmodule = sys.modules[cls.__module__]
        newcls.__module__ = cls.__module__
        setattr(clsmodule, str(cls.__name__ + '_' + name), newcls)
        setattr(newcls, '_getlinesbase', classmethod(lambda cls: baselines))
        setattr(newcls, '_getlines', classmethod(lambda cls: clslines))
        setattr(newcls, '_getlinesextrabase', classmethod(lambda cls: baseextralines))
        setattr(newcls, '_getlinesextra', classmethod(lambda cls: clsextralines))
        l2start = len(cls._getlines()) if not linesoverride else 0
        l2add = enumerate(lines2add, start=l2start)
        l2alias = {} if lalias is None else lalias._getkwargsdefault()
        for (line, linealias) in l2add:
            if not isinstance(linealias, string_types):
                linealias = linealias[0]
            desc = LineAlias(line)
            setattr(newcls, linealias, desc)
        for (line, linealias) in enumerate(newcls._getlines()):
            if not isinstance(linealias, string_types):
                linealias = linealias[0]
            desc = LineAlias(line)
            if linealias in l2alias:
                extranames = l2alias[linealias]
                if isinstance(linealias, string_types):
                    extranames = [extranames]
                for ename in extranames:
                    setattr(newcls, ename, desc)
        return newcls

    @classmethod
    def _getlinealias(cls, i):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the alias for a line given the index\n        '
        lines = cls._getlines()
        if i >= len(lines):
            return ''
        linealias = lines[i]
        return linealias

    @classmethod
    def getlinealiases(cls):
        if False:
            i = 10
            return i + 15
        return cls._getlines()

    def itersize(self):
        if False:
            print('Hello World!')
        return iter(self.lines[0:self.size()])

    def __init__(self, initlines=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create the lines recording during "_derive" or else use the\n        provided "initlines"\n        '
        self.lines = list()
        for (line, linealias) in enumerate(self._getlines()):
            kwargs = dict()
            self.lines.append(LineBuffer(**kwargs))
        for i in range(self._getlinesextra()):
            if not initlines:
                self.lines.append(LineBuffer())
            else:
                self.lines.append(initlines[i])

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Proxy line operation\n        '
        return len(self.lines[0])

    def size(self):
        if False:
            i = 10
            return i + 15
        return len(self.lines) - self._getlinesextra()

    def fullsize(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.lines)

    def extrasize(self):
        if False:
            return 10
        return self._getlinesextra()

    def __getitem__(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Proxy line operation\n        '
        return self.lines[line]

    def get(self, ago=0, size=1, line=0):
        if False:
            while True:
                i = 10
        '\n        Proxy line operation\n        '
        return self.lines[line].get(ago, size=size)

    def __setitem__(self, line, value):
        if False:
            while True:
                i = 10
        '\n        Proxy line operation\n        '
        setattr(self, self._getlinealias(line), value)

    def forward(self, value=NAN, size=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.forward(value, size=size)

    def backwards(self, size=1, force=False):
        if False:
            while True:
                i = 10
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.backwards(size, force=force)

    def rewind(self, size=1):
        if False:
            i = 10
            return i + 15
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.rewind(size)

    def extend(self, value=NAN, size=0):
        if False:
            return 10
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.extend(value, size)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.reset()

    def home(self):
        if False:
            i = 10
            return i + 15
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.home()

    def advance(self, size=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy line operation\n        '
        for line in self.lines:
            line.advance(size)

    def buflen(self, line=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy line operation\n        '
        return self.lines[line].buflen()

class MetaLineSeries(LineMultiple.__class__):
    """
    Dirty job manager for a LineSeries

      - During __new__ (class creation), it reads "lines", "plotinfo",
        "plotlines" class variable definitions and turns them into
        Classes of type Lines or AutoClassInfo (plotinfo/plotlines)

      - During "new" (instance creation) the lines/plotinfo/plotlines
        classes are substituted in the instance with instances of the
        aforementioned classes and aliases are added for the "lines" held
        in the "lines" instance

        Additionally and for remaining kwargs, these are matched against
        args in plotinfo and if existent are set there and removed from kwargs

        Remember that this Metaclass has a MetaParams (from metabase)
        as root class and therefore "params" defined for the class have been
        removed from kwargs at an earlier state
    """

    def __new__(meta, name, bases, dct):
        if False:
            i = 10
            return i + 15
        '\n        Intercept class creation, identifiy lines/plotinfo/plotlines class\n        attributes and create corresponding classes for them which take over\n        the class attributes\n        '
        aliases = dct.setdefault('alias', ())
        aliased = dct.setdefault('aliased', '')
        linesoverride = dct.pop('linesoverride', False)
        newlines = dct.pop('lines', ())
        extralines = dct.pop('extralines', 0)
        newlalias = dict(dct.pop('linealias', {}))
        newplotinfo = dict(dct.pop('plotinfo', {}))
        newplotlines = dict(dct.pop('plotlines', {}))
        cls = super(MetaLineSeries, meta).__new__(meta, name, bases, dct)
        lalias = getattr(cls, 'linealias', AutoInfoClass)
        oblalias = [x.linealias for x in bases[1:] if hasattr(x, 'linealias')]
        cls.linealias = la = lalias._derive('la_' + name, newlalias, oblalias)
        lines = getattr(cls, 'lines', Lines)
        morebaseslines = [x.lines for x in bases[1:] if hasattr(x, 'lines')]
        cls.lines = lines._derive(name, newlines, extralines, morebaseslines, linesoverride, lalias=la)
        plotinfo = getattr(cls, 'plotinfo', AutoInfoClass)
        plotlines = getattr(cls, 'plotlines', AutoInfoClass)
        morebasesplotinfo = [x.plotinfo for x in bases[1:] if hasattr(x, 'plotinfo')]
        cls.plotinfo = plotinfo._derive('pi_' + name, newplotinfo, morebasesplotinfo)
        for line in newlines:
            newplotlines.setdefault(line, dict())
        morebasesplotlines = [x.plotlines for x in bases[1:] if hasattr(x, 'plotlines')]
        cls.plotlines = plotlines._derive('pl_' + name, newplotlines, morebasesplotlines, recurse=True)
        for alias in aliases:
            newdct = {'__doc__': cls.__doc__, '__module__': cls.__module__, 'aliased': cls.__name__}
            if not isinstance(alias, string_types):
                aliasplotname = alias[1]
                alias = alias[0]
                newdct['plotinfo'] = dict(plotname=aliasplotname)
            newcls = type(str(alias), (cls,), newdct)
            clsmodule = sys.modules[cls.__module__]
            setattr(clsmodule, alias, newcls)
        return cls

    def donew(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Intercept instance creation, take over lines/plotinfo/plotlines\n        class attributes by creating corresponding instance variables and add\n        aliases for "lines" and the "lines" held within it\n        '
        plotinfo = cls.plotinfo()
        for (pname, pdef) in cls.plotinfo._getitems():
            setattr(plotinfo, pname, kwargs.pop(pname, pdef))
        (_obj, args, kwargs) = super(MetaLineSeries, cls).donew(*args, **kwargs)
        _obj.plotinfo = plotinfo
        _obj.lines = cls.lines()
        _obj.plotlines = cls.plotlines()
        _obj.l = _obj.lines
        if _obj.lines.fullsize():
            _obj.line = _obj.lines[0]
        for (l, line) in enumerate(_obj.lines):
            setattr(_obj, 'line_%s' % l, _obj._getlinealias(l))
            setattr(_obj, 'line_%d' % l, line)
            setattr(_obj, 'line%d' % l, line)
        return (_obj, args, kwargs)

class LineSeries(with_metaclass(MetaLineSeries, LineMultiple)):
    plotinfo = dict(plot=True, plotmaster=None, legendloc=None)
    csv = True

    @property
    def array(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lines[0].array

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self.lines, name)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.lines)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.lines[0][key]

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        setattr(self.lines, self.lines._getlinealias(key), value)

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(LineSeries, self).__init__()
        pass

    def plotlabel(self):
        if False:
            while True:
                i = 10
        label = self.plotinfo.plotname or self.__class__.__name__
        sublabels = self._plotlabel()
        if sublabels:
            for (i, sublabel) in enumerate(sublabels):
                if hasattr(sublabel, 'plotinfo'):
                    try:
                        s = sublabel.plotinfo.plotname
                    except:
                        s = ''
                    sublabels[i] = s or sublabel.__name__
            label += ' (%s)' % ', '.join(map(str, sublabels))
        return label

    def _plotlabel(self):
        if False:
            while True:
                i = 10
        return self.params._getvalues()

    def _getline(self, line, minusall=False):
        if False:
            while True:
                i = 10
        if isinstance(line, string_types):
            lineobj = getattr(self.lines, line)
        else:
            if line == -1:
                if minusall:
                    return None
                line = 0
            lineobj = self.lines[line]
        return lineobj

    def __call__(self, ago=None, line=-1):
        if False:
            for i in range(10):
                print('nop')
        'Returns either a delayed verison of itself in the form of a\n        LineDelay object or a timeframe adapting version with regards to a ago\n\n        Param: ago (default: None)\n\n          If ago is None or an instance of LineRoot (a lines object) the\n          returned valued is a LineCoupler instance\n\n          If ago is anything else, it is assumed to be an int and a LineDelay\n          object will be returned\n\n        Param: line (default: -1)\n          If a LinesCoupler will be returned ``-1`` means to return a\n          LinesCoupler which adapts all lines of the current LineMultiple\n          object. Else the appropriate line (referenced by name or index) will\n          be LineCoupled\n\n          If a LineDelay object will be returned, ``-1`` is the same as ``0``\n          (to retain compatibility with the previous default value of 0). This\n          behavior will change to return all existing lines in a LineDelayed\n          form\n\n          The referenced line (index or name) will be LineDelayed\n        '
        from .lineiterator import LinesCoupler
        if ago is None or isinstance(ago, LineRoot):
            args = [self, ago]
            lineobj = self._getline(line, minusall=True)
            if lineobj is not None:
                args[0] = lineobj
            return LinesCoupler(*args, _ownerskip=self)
        return LineDelay(self._getline(line), ago, _ownerskip=self)

    def forward(self, value=NAN, size=1):
        if False:
            print('Hello World!')
        self.lines.forward(value, size)

    def backwards(self, size=1, force=False):
        if False:
            for i in range(10):
                print('nop')
        self.lines.backwards(size, force=force)

    def rewind(self, size=1):
        if False:
            i = 10
            return i + 15
        self.lines.rewind(size)

    def extend(self, value=NAN, size=0):
        if False:
            i = 10
            return i + 15
        self.lines.extend(value, size)

    def reset(self):
        if False:
            print('Hello World!')
        self.lines.reset()

    def home(self):
        if False:
            print('Hello World!')
        self.lines.home()

    def advance(self, size=1):
        if False:
            print('Hello World!')
        self.lines.advance(size)

class LineSeriesStub(LineSeries):
    """Simulates a LineMultiple object based on LineSeries from a single line

    The index management operations are overriden to take into account if the
    line is a slave, ie:

      - The line reference is a line from many in a LineMultiple object
      - Both the LineMultiple object and the Line are managed by the same
        object

    Were slave not to be taken into account, the individual line would for
    example be advanced twice:

      - Once under when the LineMultiple object is advanced (because it
        advances all lines it is holding
      - Again as part of the regular management of the object holding it
    """
    extralines = 1

    def __init__(self, line, slave=False):
        if False:
            print('Hello World!')
        self.lines = self.__class__.lines(initlines=[line])
        self.owner = self._owner = line._owner
        self._minperiod = line._minperiod
        self.slave = slave

    def forward(self, value=NAN, size=1):
        if False:
            return 10
        if not self.slave:
            super(LineSeriesStub, self).forward(value, size)

    def backwards(self, size=1, force=False):
        if False:
            return 10
        if not self.slave:
            super(LineSeriesStub, self).backwards(size, force=force)

    def rewind(self, size=1):
        if False:
            print('Hello World!')
        if not self.slave:
            super(LineSeriesStub, self).rewind(size)

    def extend(self, value=NAN, size=0):
        if False:
            for i in range(10):
                print('nop')
        if not self.slave:
            super(LineSeriesStub, self).extend(value, size)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.slave:
            super(LineSeriesStub, self).reset()

    def home(self):
        if False:
            return 10
        if not self.slave:
            super(LineSeriesStub, self).home()

    def advance(self, size=1):
        if False:
            return 10
        if not self.slave:
            super(LineSeriesStub, self).advance(size)

    def qbuffer(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.slave:
            super(LineSeriesStub, self).qbuffer()

    def minbuffer(self, size):
        if False:
            while True:
                i = 10
        if not self.slave:
            super(LineSeriesStub, self).minbuffer(size)

def LineSeriesMaker(arg, slave=False):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(arg, LineSeries):
        return arg
    return LineSeriesStub(arg, slave=slave)
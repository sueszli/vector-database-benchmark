from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import itertools
import sys
import backtrader as bt
from .utils.py3 import zip, string_types, with_metaclass

def findbases(kls, topclass):
    if False:
        print('Hello World!')
    retval = list()
    for base in kls.__bases__:
        if issubclass(base, topclass):
            retval.extend(findbases(base, topclass))
            retval.append(base)
    return retval

def findowner(owned, cls, startlevel=2, skip=None):
    if False:
        return 10
    for framelevel in itertools.count(startlevel):
        try:
            frame = sys._getframe(framelevel)
        except ValueError:
            break
        self_ = frame.f_locals.get('self', None)
        if skip is not self_:
            if self_ is not owned and isinstance(self_, cls):
                return self_
        obj_ = frame.f_locals.get('_obj', None)
        if skip is not obj_:
            if obj_ is not owned and isinstance(obj_, cls):
                return obj_
    return None

class MetaBase(type):

    def doprenew(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return (cls, args, kwargs)

    def donew(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        _obj = cls.__new__(cls, *args, **kwargs)
        return (_obj, args, kwargs)

    def dopreinit(cls, _obj, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return (_obj, args, kwargs)

    def doinit(cls, _obj, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        _obj.__init__(*args, **kwargs)
        return (_obj, args, kwargs)

    def dopostinit(cls, _obj, *args, **kwargs):
        if False:
            print('Hello World!')
        return (_obj, args, kwargs)

    def __call__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        (cls, args, kwargs) = cls.doprenew(*args, **kwargs)
        (_obj, args, kwargs) = cls.donew(*args, **kwargs)
        (_obj, args, kwargs) = cls.dopreinit(_obj, *args, **kwargs)
        (_obj, args, kwargs) = cls.doinit(_obj, *args, **kwargs)
        (_obj, args, kwargs) = cls.dopostinit(_obj, *args, **kwargs)
        return _obj

class AutoInfoClass(object):
    _getpairsbase = classmethod(lambda cls: OrderedDict())
    _getpairs = classmethod(lambda cls: OrderedDict())
    _getrecurse = classmethod(lambda cls: False)

    @classmethod
    def _derive(cls, name, info, otherbases, recurse=False):
        if False:
            for i in range(10):
                print('nop')
        baseinfo = cls._getpairs().copy()
        obasesinfo = OrderedDict()
        for obase in otherbases:
            if isinstance(obase, (tuple, dict)):
                obasesinfo.update(obase)
            else:
                obasesinfo.update(obase._getpairs())
        baseinfo.update(obasesinfo)
        clsinfo = baseinfo.copy()
        clsinfo.update(info)
        info2add = obasesinfo.copy()
        info2add.update(info)
        clsmodule = sys.modules[cls.__module__]
        newclsname = str(cls.__name__ + '_' + name)
        namecounter = 1
        while hasattr(clsmodule, newclsname):
            newclsname += str(namecounter)
            namecounter += 1
        newcls = type(newclsname, (cls,), {})
        setattr(clsmodule, newclsname, newcls)
        setattr(newcls, '_getpairsbase', classmethod(lambda cls: baseinfo.copy()))
        setattr(newcls, '_getpairs', classmethod(lambda cls: clsinfo.copy()))
        setattr(newcls, '_getrecurse', classmethod(lambda cls: recurse))
        for (infoname, infoval) in info2add.items():
            if recurse:
                recursecls = getattr(newcls, infoname, AutoInfoClass)
                infoval = recursecls._derive(name + '_' + infoname, infoval, [])
            setattr(newcls, infoname, infoval)
        return newcls

    def isdefault(self, pname):
        if False:
            i = 10
            return i + 15
        return self._get(pname) == self._getkwargsdefault()[pname]

    def notdefault(self, pname):
        if False:
            i = 10
            return i + 15
        return self._get(pname) != self._getkwargsdefault()[pname]

    def _get(self, name, default=None):
        if False:
            while True:
                i = 10
        return getattr(self, name, default)

    @classmethod
    def _getkwargsdefault(cls):
        if False:
            i = 10
            return i + 15
        return cls._getpairs()

    @classmethod
    def _getkeys(cls):
        if False:
            print('Hello World!')
        return cls._getpairs().keys()

    @classmethod
    def _getdefaults(cls):
        if False:
            for i in range(10):
                print('nop')
        return list(cls._getpairs().values())

    @classmethod
    def _getitems(cls):
        if False:
            i = 10
            return i + 15
        return cls._getpairs().items()

    @classmethod
    def _gettuple(cls):
        if False:
            while True:
                i = 10
        return tuple(cls._getpairs().items())

    def _getkwargs(self, skip_=False):
        if False:
            print('Hello World!')
        l = [(x, getattr(self, x)) for x in self._getkeys() if not skip_ or not x.startswith('_')]
        return OrderedDict(l)

    def _getvalues(self):
        if False:
            i = 10
            return i + 15
        return [getattr(self, x) for x in self._getkeys()]

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        obj = super(AutoInfoClass, cls).__new__(cls, *args, **kwargs)
        if cls._getrecurse():
            for infoname in obj._getkeys():
                recursecls = getattr(cls, infoname)
                setattr(obj, infoname, recursecls())
        return obj

class MetaParams(MetaBase):

    def __new__(meta, name, bases, dct):
        if False:
            return 10
        newparams = dct.pop('params', ())
        packs = 'packages'
        newpackages = tuple(dct.pop(packs, ()))
        fpacks = 'frompackages'
        fnewpackages = tuple(dct.pop(fpacks, ()))
        cls = super(MetaParams, meta).__new__(meta, name, bases, dct)
        params = getattr(cls, 'params', AutoInfoClass)
        packages = tuple(getattr(cls, packs, ()))
        fpackages = tuple(getattr(cls, fpacks, ()))
        morebasesparams = [x.params for x in bases[1:] if hasattr(x, 'params')]
        for y in [x.packages for x in bases[1:] if hasattr(x, packs)]:
            packages += tuple(y)
        for y in [x.frompackages for x in bases[1:] if hasattr(x, fpacks)]:
            fpackages += tuple(y)
        cls.packages = packages + newpackages
        cls.frompackages = fpackages + fnewpackages
        cls.params = params._derive(name, newparams, morebasesparams)
        return cls

    def donew(cls, *args, **kwargs):
        if False:
            return 10
        clsmod = sys.modules[cls.__module__]
        for p in cls.packages:
            if isinstance(p, (tuple, list)):
                (p, palias) = p
            else:
                palias = p
            pmod = __import__(p)
            plevels = p.split('.')
            if p == palias and len(plevels) > 1:
                setattr(clsmod, pmod.__name__, pmod)
            else:
                for plevel in plevels[1:]:
                    pmod = getattr(pmod, plevel)
                setattr(clsmod, palias, pmod)
        for (p, frompackage) in cls.frompackages:
            if isinstance(frompackage, string_types):
                frompackage = (frompackage,)
            for fp in frompackage:
                if isinstance(fp, (tuple, list)):
                    (fp, falias) = fp
                else:
                    (fp, falias) = (fp, fp)
                pmod = __import__(p, fromlist=[str(fp)])
                pattr = getattr(pmod, fp)
                setattr(clsmod, falias, pattr)
                for basecls in cls.__bases__:
                    setattr(sys.modules[basecls.__module__], falias, pattr)
        params = cls.params()
        for (pname, pdef) in cls.params._getitems():
            setattr(params, pname, kwargs.pop(pname, pdef))
        (_obj, args, kwargs) = super(MetaParams, cls).donew(*args, **kwargs)
        _obj.params = params
        _obj.p = params
        return (_obj, args, kwargs)

class ParamsBase(with_metaclass(MetaParams, object)):
    pass

class ItemCollection(object):
    """
    Holds a collection of items that can be reached by

      - Index
      - Name (if set in the append operation)
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._items = list()
        self._names = list()

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._items)

    def append(self, item, name=None):
        if False:
            print('Hello World!')
        setattr(self, name, item)
        self._items.append(item)
        if name:
            self._names.append(name)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._items[key]

    def getnames(self):
        if False:
            print('Hello World!')
        return self._names

    def getitems(self):
        if False:
            i = 10
            return i + 15
        return zip(self._names, self._items)

    def getbyname(self, name):
        if False:
            for i in range(10):
                print('nop')
        idx = self._names.index(name)
        return self._items[idx]
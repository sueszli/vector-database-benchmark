from __future__ import unicode_literals, division, absolute_import, print_function
from powerline.lib.unicode import unicode

def gen_new(cls):
    if False:
        print('Hello World!')

    def __new__(arg_cls, value, mark):
        if False:
            i = 10
            return i + 15
        r = super(arg_cls, arg_cls).__new__(arg_cls, value)
        r.mark = mark
        r.value = value
        return r
    return __new__

def gen_init(cls):
    if False:
        for i in range(10):
            print('nop')

    def __init__(self, value, mark):
        if False:
            i = 10
            return i + 15
        return cls.__init__(self, value)
    return __init__

def gen_getnewargs(cls):
    if False:
        while True:
            i = 10

    def __getnewargs__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.value, self.mark)
    return __getnewargs__

class MarkedUnicode(unicode):
    __new__ = gen_new(unicode)
    __getnewargs__ = gen_getnewargs(unicode)

    def _proc_partition(self, part_result):
        if False:
            i = 10
            return i + 15
        pointdiff = 1
        r = []
        for s in part_result:
            r.append(MarkedUnicode(s, self.mark.advance_string(pointdiff)))
            pointdiff += len(s)
        return tuple(r)

    def rpartition(self, sep):
        if False:
            for i in range(10):
                print('nop')
        return self._proc_partition(super(MarkedUnicode, self).rpartition(sep))

    def partition(self, sep):
        if False:
            print('Hello World!')
        return self._proc_partition(super(MarkedUnicode, self).partition(sep))

class MarkedInt(int):
    __new__ = gen_new(int)
    __getnewargs__ = gen_getnewargs(int)

class MarkedFloat(float):
    __new__ = gen_new(float)
    __getnewargs__ = gen_getnewargs(float)

class MarkedDict(dict):
    __init__ = gen_init(dict)
    __getnewargs__ = gen_getnewargs(dict)

    def __new__(arg_cls, value, mark):
        if False:
            return 10
        r = super(arg_cls, arg_cls).__new__(arg_cls, value)
        r.mark = mark
        r.value = value
        r.keydict = dict(((key, key) for key in r))
        return r

    def setmerged(self, d):
        if False:
            i = 10
            return i + 15
        try:
            self.mark.set_merged_mark(d.mark)
        except AttributeError:
            pass

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        try:
            old_value = self[key]
        except KeyError:
            pass
        else:
            try:
                key.mark.set_old_mark(self.keydict[key].mark)
            except AttributeError:
                pass
            except KeyError:
                pass
            try:
                value.mark.set_old_mark(old_value.mark)
            except AttributeError:
                pass
        dict.__setitem__(self, key, value)
        self.keydict[key] = key

    def update(self, *args, **kwargs):
        if False:
            print('Hello World!')
        dict.update(self, *args, **kwargs)
        self.keydict = dict(((key, key) for key in self))

    def copy(self):
        if False:
            print('Hello World!')
        return MarkedDict(super(MarkedDict, self).copy(), self.mark)

class MarkedList(list):
    __new__ = gen_new(list)
    __init__ = gen_init(list)
    __getnewargs__ = gen_getnewargs(list)

class MarkedValue:

    def __init__(self, value, mark):
        if False:
            for i in range(10):
                print('nop')
        self.mark = mark
        self.value = value
    __getinitargs__ = gen_getnewargs(None)
specialclasses = {unicode: MarkedUnicode, int: MarkedInt, float: MarkedFloat, dict: MarkedDict, list: MarkedList}
classcache = {}

def gen_marked_value(value, mark, use_special_classes=True):
    if False:
        return 10
    if use_special_classes and value.__class__ in specialclasses:
        Marked = specialclasses[value.__class__]
    elif value.__class__ in classcache:
        Marked = classcache[value.__class__]
    else:

        class Marked(MarkedValue):
            for func in value.__class__.__dict__:
                if func == 'copy':

                    def copy(self):
                        if False:
                            print('Hello World!')
                        return self.__class__(self.value.copy(), self.mark)
                elif func not in set(('__init__', '__new__', '__getattribute__')):
                    if func in set(('__eq__',)):
                        exec('def {0}(self, *args):\n\treturn self.value.{0}(*[arg.value if isinstance(arg, MarkedValue) else arg for arg in args])'.format(func))
                    else:
                        exec('def {0}(self, *args, **kwargs):\n\treturn self.value.{0}(*args, **kwargs)\n'.format(func))
        classcache[value.__class__] = Marked
    return Marked(value, mark)
import sys, copy, types as pytypes
IS_RPYTHON = sys.argv[0].endswith('rpython')
if IS_RPYTHON:
    from rpython.rlib.listsort import TimSort
else:
    import re

class StringSort(TimSort):

    def lt(self, a, b):
        if False:
            while True:
                i = 10
        assert isinstance(a, unicode)
        assert isinstance(b, unicode)
        return a < b

def _equal_Q(a, b):
    if False:
        while True:
            i = 10
    assert isinstance(a, MalType) and isinstance(b, MalType)
    (ota, otb) = (a.__class__, b.__class__)
    if not (ota is otb or (_sequential_Q(a) and _sequential_Q(b))):
        return False
    if isinstance(a, MalSym) and isinstance(b, MalSym):
        return a.value == b.value
    elif isinstance(a, MalStr) and isinstance(b, MalStr):
        return a.value == b.value
    elif isinstance(a, MalInt) and isinstance(b, MalInt):
        return a.value == b.value
    elif _list_Q(a) or _vector_Q(a):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not _equal_Q(a[i], b[i]):
                return False
        return True
    elif _hash_map_Q(a):
        assert isinstance(a, MalHashMap)
        assert isinstance(b, MalHashMap)
        akeys = a.dct.keys()
        bkeys = b.dct.keys()
        if len(akeys) != len(bkeys):
            return False
        StringSort(akeys).sort()
        StringSort(bkeys).sort()
        for i in range(len(akeys)):
            (ak, bk) = (akeys[i], bkeys[i])
            assert isinstance(ak, unicode)
            assert isinstance(bk, unicode)
            if ak != bk:
                return False
            (av, bv) = (a.dct[ak], b.dct[bk])
            if not _equal_Q(av, bv):
                return False
        return True
    elif a is b:
        return True
    else:
        throw_str('no = op defined for %s' % a.__class__.__name__)

def _sequential_Q(seq):
    if False:
        i = 10
        return i + 15
    return _list_Q(seq) or _vector_Q(seq)

def _clone(obj):
    if False:
        print('Hello World!')
    if isinstance(obj, MalFunc):
        return MalFunc(obj.fn, obj.ast, obj.env, obj.params, obj.EvalFunc, obj.ismacro)
    elif isinstance(obj, MalList):
        return obj.__class__(obj.values)
    elif isinstance(obj, MalHashMap):
        return MalHashMap(obj.dct)
    elif isinstance(obj, MalAtom):
        return MalAtom(obj.value)
    else:
        raise Exception('_clone on invalid type')

def _replace(match, sub, old_str):
    if False:
        print('Hello World!')
    new_str = u''
    idx = 0
    while idx < len(old_str):
        midx = old_str.find(match, idx)
        if midx < 0:
            break
        assert midx >= 0 and midx < len(old_str)
        new_str = new_str + old_str[idx:midx]
        new_str = new_str + sub
        idx = midx + len(match)
    new_str = new_str + old_str[idx:]
    return new_str

class MalException(Exception):

    def __init__(self, object):
        if False:
            i = 10
            return i + 15
        self.object = object

def throw_str(s):
    if False:
        i = 10
        return i + 15
    raise MalException(MalStr(unicode(s)))

class MalType:
    pass

class MalMeta(MalType):
    pass

class MalNil(MalType):
    pass
nil = MalNil()

def _nil_Q(exp):
    if False:
        i = 10
        return i + 15
    assert isinstance(exp, MalType)
    return exp is nil

class MalTrue(MalType):
    pass
true = MalTrue()

def _true_Q(exp):
    if False:
        print('Hello World!')
    assert isinstance(exp, MalType)
    return exp is true

class MalFalse(MalType):
    pass
false = MalFalse()

def _false_Q(exp):
    if False:
        while True:
            i = 10
    assert isinstance(exp, MalType)
    return exp is false

class MalInt(MalType):

    def __init__(self, value):
        if False:
            print('Hello World!')
        assert isinstance(value, int)
        self.value = value

def _int_Q(exp):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(exp, MalType)
    return exp.__class__ is MalInt

class MalStr(MalType):

    def __init__(self, value):
        if False:
            return 10
        assert isinstance(value, unicode)
        self.value = value

    def __len__(self):
        if False:
            return 10
        return len(self.value)

def _string_Q(exp):
    if False:
        i = 10
        return i + 15
    assert isinstance(exp, MalType)
    return exp.__class__ is MalStr and (not _keyword_Q(exp))

def _keyword(mstr):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(mstr, MalType)
    if isinstance(mstr, MalStr):
        val = mstr.value
        if val[0] == u'ʞ':
            return mstr
        else:
            return MalStr(u'ʞ' + val)
    else:
        throw_str('_keyword called on non-string')

def _keywordu(strn):
    if False:
        while True:
            i = 10
    assert isinstance(strn, unicode)
    return MalStr(u'ʞ' + strn)

def _keyword_Q(exp):
    if False:
        return 10
    if isinstance(exp, MalStr) and len(exp.value) > 0:
        return exp.value[0] == u'ʞ'
    else:
        return False

class MalSym(MalMeta):

    def __init__(self, value):
        if False:
            print('Hello World!')
        assert isinstance(value, unicode)
        self.value = value
        self.meta = nil

def _symbol(strn):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(strn, unicode)
    return MalSym(strn)

def _symbol_Q(exp):
    if False:
        print('Hello World!')
    assert isinstance(exp, MalType)
    return exp.__class__ is MalSym

class MalList(MalMeta):

    def __init__(self, vals):
        if False:
            i = 10
            return i + 15
        assert isinstance(vals, list)
        self.values = vals
        self.meta = nil

    def append(self, val):
        if False:
            return 10
        self.values.append(val)

    def rest(self):
        if False:
            return 10
        return MalList(self.values[1:])

    def __len__(self):
        if False:
            return 10
        return len(self.values)

    def __getitem__(self, i):
        if False:
            return 10
        assert isinstance(i, int)
        return self.values[i]

    def slice(self, start):
        if False:
            while True:
                i = 10
        return MalList(self.values[start:len(self.values)])

    def slice2(self, start, end):
        if False:
            print('Hello World!')
        assert end >= 0
        return MalList(self.values[start:end])

def _list(*vals):
    if False:
        return 10
    return MalList(list(vals))

def _listl(lst):
    if False:
        for i in range(10):
            print('nop')
    return MalList(lst)

def _list_Q(exp):
    if False:
        return 10
    assert isinstance(exp, MalType)
    return exp.__class__ is MalList

class MalVector(MalList):
    pass

def _vector(*vals):
    if False:
        for i in range(10):
            print('nop')
    return MalVector(list(vals))

def _vectorl(lst):
    if False:
        for i in range(10):
            print('nop')
    return MalVector(lst)

def _vector_Q(exp):
    if False:
        print('Hello World!')
    assert isinstance(exp, MalType)
    return exp.__class__ is MalVector

class MalHashMap(MalMeta):

    def __init__(self, dct):
        if False:
            while True:
                i = 10
        self.dct = dct
        self.meta = nil

    def append(self, val):
        if False:
            i = 10
            return i + 15
        self.dct.append(val)

    def __getitem__(self, k):
        if False:
            return 10
        assert isinstance(k, unicode)
        if not isinstance(k, unicode):
            throw_str('hash-map lookup by non-string/non-keyword')
        return self.dct[k]

    def __setitem__(self, k, v):
        if False:
            while True:
                i = 10
        if not isinstance(k, unicode):
            throw_str('hash-map key must be string or keyword')
        assert isinstance(v, MalType)
        self.dct[k] = v
        return v

def _hash_mapl(kvs):
    if False:
        print('Hello World!')
    dct = {}
    for i in range(0, len(kvs), 2):
        k = kvs[i]
        if not isinstance(k, MalStr):
            throw_str('hash-map key must be string or keyword')
        v = kvs[i + 1]
        dct[k.value] = v
    return MalHashMap(dct)

def _hash_map_Q(exp):
    if False:
        print('Hello World!')
    assert isinstance(exp, MalType)
    return exp.__class__ is MalHashMap
from env import Env

class MalFunc(MalMeta):

    def __init__(self, fn, ast=None, env=None, params=None, EvalFunc=None, ismacro=False):
        if False:
            while True:
                i = 10
        if fn is None and EvalFunc is None:
            throw_str('MalFunc requires either fn or EvalFunc')
        self.fn = fn
        self.ast = ast
        self.env = env
        self.params = params
        self.EvalFunc = EvalFunc
        self.ismacro = ismacro
        self.meta = nil

    def apply(self, args):
        if False:
            return 10
        if self.EvalFunc:
            return self.EvalFunc(self.ast, self.gen_env(args))
        else:
            return self.fn(args)

    def gen_env(self, args):
        if False:
            print('Hello World!')
        return Env(self.env, self.params, args)

def _function_Q(exp):
    if False:
        i = 10
        return i + 15
    assert isinstance(exp, MalType)
    return exp.__class__ is MalFunc

class MalAtom(MalMeta):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value
        self.meta = nil

    def get_value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

def _atom(val):
    if False:
        while True:
            i = 10
    return MalAtom(val)

def _atom_Q(exp):
    if False:
        while True:
            i = 10
    return exp.__class__ is MalAtom
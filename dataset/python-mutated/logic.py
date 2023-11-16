"""Logic expressions handling

NOTE
----

at present this is mainly needed for facts.py, feel free however to improve
this stuff for general purpose.
"""
from __future__ import annotations
from typing import Optional
FuzzyBool = Optional[bool]

def _torf(args):
    if False:
        return 10
    'Return True if all args are True, False if they\n    are all False, else None.\n\n    >>> from sympy.core.logic import _torf\n    >>> _torf((True, True))\n    True\n    >>> _torf((False, False))\n    False\n    >>> _torf((True, False))\n    '
    sawT = sawF = False
    for a in args:
        if a is True:
            if sawF:
                return
            sawT = True
        elif a is False:
            if sawT:
                return
            sawF = True
        else:
            return
    return sawT

def _fuzzy_group(args, quick_exit=False):
    if False:
        return 10
    'Return True if all args are True, None if there is any None else False\n    unless ``quick_exit`` is True (then return None as soon as a second False\n    is seen.\n\n     ``_fuzzy_group`` is like ``fuzzy_and`` except that it is more\n    conservative in returning a False, waiting to make sure that all\n    arguments are True or False and returning None if any arguments are\n    None. It also has the capability of permiting only a single False and\n    returning None if more than one is seen. For example, the presence of a\n    single transcendental amongst rationals would indicate that the group is\n    no longer rational; but a second transcendental in the group would make the\n    determination impossible.\n\n\n    Examples\n    ========\n\n    >>> from sympy.core.logic import _fuzzy_group\n\n    By default, multiple Falses mean the group is broken:\n\n    >>> _fuzzy_group([False, False, True])\n    False\n\n    If multiple Falses mean the group status is unknown then set\n    `quick_exit` to True so None can be returned when the 2nd False is seen:\n\n    >>> _fuzzy_group([False, False, True], quick_exit=True)\n\n    But if only a single False is seen then the group is known to\n    be broken:\n\n    >>> _fuzzy_group([False, True, True], quick_exit=True)\n    False\n\n    '
    saw_other = False
    for a in args:
        if a is True:
            continue
        if a is None:
            return
        if quick_exit and saw_other:
            return
        saw_other = True
    return not saw_other

def fuzzy_bool(x):
    if False:
        print('Hello World!')
    'Return True, False or None according to x.\n\n    Whereas bool(x) returns True or False, fuzzy_bool allows\n    for the None value and non-false values (which become None), too.\n\n    Examples\n    ========\n\n    >>> from sympy.core.logic import fuzzy_bool\n    >>> from sympy.abc import x\n    >>> fuzzy_bool(x), fuzzy_bool(None)\n    (None, None)\n    >>> bool(x), bool(None)\n    (True, False)\n\n    '
    if x is None:
        return None
    if x in (True, False):
        return bool(x)

def fuzzy_and(args):
    if False:
        i = 10
        return i + 15
    'Return True (all True), False (any False) or None.\n\n    Examples\n    ========\n\n    >>> from sympy.core.logic import fuzzy_and\n    >>> from sympy import Dummy\n\n    If you had a list of objects to test the commutivity of\n    and you want the fuzzy_and logic applied, passing an\n    iterator will allow the commutativity to only be computed\n    as many times as necessary. With this list, False can be\n    returned after analyzing the first symbol:\n\n    >>> syms = [Dummy(commutative=False), Dummy()]\n    >>> fuzzy_and(s.is_commutative for s in syms)\n    False\n\n    That False would require less work than if a list of pre-computed\n    items was sent:\n\n    >>> fuzzy_and([s.is_commutative for s in syms])\n    False\n    '
    rv = True
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is False:
            return False
        if rv:
            rv = ai
    return rv

def fuzzy_not(v):
    if False:
        return 10
    '\n    Not in fuzzy logic\n\n    Return None if `v` is None else `not v`.\n\n    Examples\n    ========\n\n    >>> from sympy.core.logic import fuzzy_not\n    >>> fuzzy_not(True)\n    False\n    >>> fuzzy_not(None)\n    >>> fuzzy_not(False)\n    True\n\n    '
    if v is None:
        return v
    else:
        return not v

def fuzzy_or(args):
    if False:
        print('Hello World!')
    "\n    Or in fuzzy logic. Returns True (any True), False (all False), or None\n\n    See the docstrings of fuzzy_and and fuzzy_not for more info.  fuzzy_or is\n    related to the two by the standard De Morgan's law.\n\n    >>> from sympy.core.logic import fuzzy_or\n    >>> fuzzy_or([True, False])\n    True\n    >>> fuzzy_or([True, None])\n    True\n    >>> fuzzy_or([False, False])\n    False\n    >>> print(fuzzy_or([False, None]))\n    None\n\n    "
    rv = False
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is True:
            return True
        if rv is False:
            rv = ai
    return rv

def fuzzy_xor(args):
    if False:
        while True:
            i = 10
    'Return None if any element of args is not True or False, else\n    True (if there are an odd number of True elements), else False.'
    t = f = 0
    for a in args:
        ai = fuzzy_bool(a)
        if ai:
            t += 1
        elif ai is False:
            f += 1
        else:
            return
    return t % 2 == 1

def fuzzy_nand(args):
    if False:
        i = 10
        return i + 15
    'Return False if all args are True, True if they are all False,\n    else None.'
    return fuzzy_not(fuzzy_and(args))

class Logic:
    """Logical expression"""
    op_2class: dict[str, type[Logic]] = {}

    def __new__(cls, *args):
        if False:
            print('Hello World!')
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __getnewargs__(self):
        if False:
            while True:
                i = 10
        return self.args

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(a, b):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(b, type(a)):
            return False
        else:
            return a.args == b.args

    def __ne__(a, b):
        if False:
            i = 10
            return i + 15
        if not isinstance(b, type(a)):
            return True
        else:
            return a.args != b.args

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if self.__cmp__(other) == -1:
            return True
        return False

    def __cmp__(self, other):
        if False:
            while True:
                i = 10
        if type(self) is not type(other):
            a = str(type(self))
            b = str(type(other))
        else:
            a = self.args
            b = other.args
        return (a > b) - (a < b)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s(%s)' % (self.__class__.__name__, ', '.join((str(a) for a in self.args)))
    __repr__ = __str__

    @staticmethod
    def fromstring(text):
        if False:
            return 10
        'Logic from string with space around & and | but none after !.\n\n           e.g.\n\n           !a & b | c\n        '
        lexpr = None
        schedop = None
        for term in text.split():
            if term in '&|':
                if schedop is not None:
                    raise ValueError('double op forbidden: "%s %s"' % (term, schedop))
                if lexpr is None:
                    raise ValueError('%s cannot be in the beginning of expression' % term)
                schedop = term
                continue
            if '&' in term or '|' in term:
                raise ValueError('& and | must have space around them')
            if term[0] == '!':
                if len(term) == 1:
                    raise ValueError('do not include space after "!"')
                term = Not(term[1:])
            if schedop:
                lexpr = Logic.op_2class[schedop](lexpr, term)
                schedop = None
                continue
            if lexpr is not None:
                raise ValueError('missing op between "%s" and "%s"' % (lexpr, term))
            lexpr = term
        if schedop is not None:
            raise ValueError('premature end-of-expression in "%s"' % text)
        if lexpr is None:
            raise ValueError('"%s" is empty' % text)
        return lexpr

class AndOr_Base(Logic):

    def __new__(cls, *args):
        if False:
            return 10
        bargs = []
        for a in args:
            if a == cls.op_x_notx:
                return a
            elif a == (not cls.op_x_notx):
                continue
            bargs.append(a)
        args = sorted(set(cls.flatten(bargs)), key=hash)
        for a in args:
            if Not(a) in args:
                return cls.op_x_notx
        if len(args) == 1:
            return args.pop()
        elif len(args) == 0:
            return not cls.op_x_notx
        return Logic.__new__(cls, *args)

    @classmethod
    def flatten(cls, args):
        if False:
            while True:
                i = 10
        args_queue = list(args)
        res = []
        while True:
            try:
                arg = args_queue.pop(0)
            except IndexError:
                break
            if isinstance(arg, Logic):
                if isinstance(arg, cls):
                    args_queue.extend(arg.args)
                    continue
            res.append(arg)
        args = tuple(res)
        return args

class And(AndOr_Base):
    op_x_notx = False

    def _eval_propagate_not(self):
        if False:
            for i in range(10):
                print('nop')
        return Or(*[Not(a) for a in self.args])

    def expand(self):
        if False:
            for i in range(10):
                print('nop')
        for (i, arg) in enumerate(self.args):
            if isinstance(arg, Or):
                arest = self.args[:i] + self.args[i + 1:]
                orterms = [And(*arest + (a,)) for a in arg.args]
                for j in range(len(orterms)):
                    if isinstance(orterms[j], Logic):
                        orterms[j] = orterms[j].expand()
                res = Or(*orterms)
                return res
        return self

class Or(AndOr_Base):
    op_x_notx = True

    def _eval_propagate_not(self):
        if False:
            for i in range(10):
                print('nop')
        return And(*[Not(a) for a in self.args])

class Not(Logic):

    def __new__(cls, arg):
        if False:
            return 10
        if isinstance(arg, str):
            return Logic.__new__(cls, arg)
        elif isinstance(arg, bool):
            return not arg
        elif isinstance(arg, Not):
            return arg.args[0]
        elif isinstance(arg, Logic):
            arg = arg._eval_propagate_not()
            return arg
        else:
            raise ValueError('Not: unknown argument %r' % (arg,))

    @property
    def arg(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]
Logic.op_2class['&'] = And
Logic.op_2class['|'] = Or
Logic.op_2class['!'] = Not
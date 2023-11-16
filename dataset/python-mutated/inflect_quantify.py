from __future__ import unicode_literals
from __future__ import division
from builtins import str, bytes, dict, int
from builtins import map, zip, filter
from builtins import object, range
import os
import sys
import re
from math import log, ceil
try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except:
    MODULE = ''
sys.path.insert(0, os.path.join(MODULE, '..', '..', '..', '..'))
from pattern.text.en.inflect import pluralize, referenced
sys.path.pop(0)
NUMERALS = {'zero': 0, 'ten': 10, 'twenty': 20, 'one': 1, 'eleven': 11, 'thirty': 30, 'two': 2, 'twelve': 12, 'forty': 40, 'three': 3, 'thirteen': 13, 'fifty': 50, 'four': 4, 'fourteen': 14, 'sixty': 60, 'five': 5, 'fifteen': 15, 'seventy': 70, 'six': 6, 'sixteen': 16, 'eighty': 80, 'seven': 7, 'seventeen': 17, 'ninety': 90, 'eight': 8, 'eighteen': 18, 'nine': 9, 'nineteen': 19}
NUMERALS_INVERSE = dict(((i, w) for (w, i) in NUMERALS.items()))
NUMERALS_VERBOSE = {'half': (1, 0.5), 'dozen': (12, 0.0), 'score': (20, 0.0)}
ORDER = ['hundred', 'thousand'] + [m + 'illion' for m in ('m', 'b', 'tr', 'quadr', 'quint', 'sext', 'sept', 'oct', 'non', 'dec', 'undec', 'duodec', 'tredec', 'quattuordec', 'quindec', 'sexdec', 'septemdec', 'octodec', 'novemdec', 'vigint')]
O = {ORDER[0]: 100, ORDER[1]: 1000}
for (i, k) in enumerate(ORDER[2:]):
    O[k] = 1000000 * 1000 ** i
(ZERO, MINUS, RADIX, THOUSANDS, CONJUNCTION) = ('zero', 'minus', 'point', ',', 'and')

def zshift(s):
    if False:
        return 10
    ' Returns a (string, count)-tuple, with leading zeros strippped from the string and counted.\n    '
    s = s.lstrip()
    i = 0
    while s.startswith((ZERO, '0')):
        s = re.sub('^(0|%s)\\s*' % ZERO, '', s, 1)
        i = i + 1
    return (s, i)

def number(s):
    if False:
        while True:
            i = 10
    ' Returns the given numeric string as a float or an int.\n        If no number can be parsed from the string, returns 0.\n        For example:\n        number("five point two million") => 5200000\n        number("seventy-five point two") => 75.2\n        number("three thousand and one") => 3001\n    '
    s = s.strip()
    s = s.lower()
    if s.startswith(MINUS):
        return -number(s.replace(MINUS, '', 1))
    s = s.replace('&', ' %s ' % CONJUNCTION)
    s = s.replace(THOUSANDS, '')
    s = s.replace('-', ' ')
    s = s.split(RADIX)
    if len(s) > 1:
        f = ' '.join(s[1:])
        (f, z) = zshift(f)
        f = float(number(f))
        f /= 10 ** (len(str(int(f))) + z)
    else:
        f = 0
    i = n = 0
    s = s[0].split()
    for (j, x) in enumerate(s):
        if x in NUMERALS:
            i += NUMERALS[x]
        elif x in NUMERALS_VERBOSE:
            i = i * NUMERALS_VERBOSE[x][0] + NUMERALS_VERBOSE[x][1]
        elif x in O:
            i *= O[x]
            if j < len(s) - 1 and s[j + 1] in O:
                continue
            if O[x] > 100:
                n += i
                i = 0
        elif x == CONJUNCTION:
            pass
        else:
            try:
                i += '.' in x and float(x) or int(x)
            except:
                pass
    return n + i + f

def numerals(n, round=2):
    if False:
        return 10
    ' Returns the given int or float as a string of numerals.\n        By default, the fractional part is rounded to two decimals.\n        For example:\n        numerals(4011) => four thousand and eleven\n        numerals(2.25) => two point twenty-five\n        numerals(2.249) => two point twenty-five\n        numerals(2.249, round=3) => two point two hundred and forty-nine\n    '
    if isinstance(n, str):
        if n.isdigit():
            n = int(n)
        else:
            if round is None:
                round = len(n.split('.')[1])
            n = float(n)
    if n < 0:
        return '%s %s' % (MINUS, numerals(abs(n)))
    i = int(n // 1)
    f = n - i
    r = 0
    if i in NUMERALS_INVERSE:
        s = NUMERALS_INVERSE[i]
    elif i < 100:
        s = numerals(i // 10 * 10) + '-' + numerals(i % 10)
    elif i < 1000:
        s = numerals(i // 100) + ' ' + ORDER[0]
        r = i % 100
    else:
        s = ''
        (o, base) = (1, 1000)
        while i > base:
            o += 1
            base *= 1000
        while o > len(ORDER) - 1:
            s += ' ' + ORDER[-1]
            o -= len(ORDER) - 1
        s = '%s %s%s' % (numerals(i // int(base / 1000)), o > 1 and ORDER[o - 1] or '', s)
        r = i % (base / 1000)
    if f != 0:
        f = ('%.' + str(round is None and 2 or round) + 'f') % f
        f = f.replace('0.', '', 1).rstrip('0')
        (f, z) = zshift(f)
        f = f and ' %s%s %s' % (RADIX, ' %s' % ZERO * z, numerals(int(f))) or ''
    else:
        f = ''
    if r == 0:
        return s + f
    elif r >= 1000:
        return '%s%s %s' % (s, THOUSANDS, numerals(r) + f)
    elif r <= 100:
        return '%s %s %s' % (s, CONJUNCTION, numerals(r) + f)
    else:
        return '%s %s' % (s, numerals(r) + f)
NONE = 'no'
PAIR = 'a pair of'
SEVERAL = 'several'
NUMBER = 'a number of'
SCORE = 'a score of'
DOZENS = 'dozens of'
COUNTLESS = 'countless'
quantify_custom_plurals = {}

def approximate(word, amount=1, plural={}):
    if False:
        print('Hello World!')
    ' Returns an approximation of the number of given objects.\n        Two objects are described as being "a pair",\n        smaller than eight is "several",\n        smaller than twenty is "a number of",\n        smaller than two hundred are "dozens",\n        anything bigger is described as being tens or hundreds of thousands or millions.\n        For example: approximate("chicken", 100) => "dozens of chickens".\n    '
    try:
        p = pluralize(word, custom=plural)
    except:
        raise TypeError("can't pluralize %s (not a string)" % word.__class__.__name__)
    if amount == 0:
        return '%s %s' % (NONE, p)
    if amount == 1:
        return referenced(word)
    if amount == 2:
        return '%s %s' % (PAIR, p)
    if 3 <= amount < 8:
        return '%s %s' % (SEVERAL, p)
    if 8 <= amount < 18:
        return '%s %s' % (NUMBER, p)
    if 18 <= amount < 23:
        return '%s %s' % (SCORE, p)
    if 23 <= amount < 200:
        return '%s %s' % (DOZENS, p)
    if amount > 10000000:
        return '%s %s' % (COUNTLESS, p)
    thousands = int(log(amount, 10) / 3)
    hundreds = ceil(log(amount, 10) % 3) - 1
    h = hundreds == 2 and 'hundreds of ' or (hundreds == 1 and 'tens of ' or '')
    t = thousands > 0 and pluralize(ORDER[thousands]) + ' of ' or ''
    return '%s%s%s' % (h, t, p)

def count(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' Returns an approximation of the entire set.\n        Identical words are grouped and counted and then quantified with an approximation.\n    '
    if len(args) == 2 and isinstance(args[0], str):
        return approximate(args[0], args[1], kwargs.get('plural', {}))
    if len(args) == 1 and isinstance(args[0], str) and ('amount' in kwargs):
        return approximate(args[0], kwargs['amount'], kwargs.get('plural', {}))
    if len(args) == 1 and isinstance(args[0], dict):
        count = args[0]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        count = {}
        for word in args[0]:
            try:
                count.setdefault(word, 0)
                count[word] += 1
            except:
                raise TypeError("can't count %s (not a string)" % word.__class__.__name__)
    s = [(count[word], word) for word in count]
    s = max([n for (n, w) in s]) > 1 and reversed(sorted(s)) or s
    phrase = []
    for (i, (n, word)) in enumerate(s):
        phrase.append(approximate(word, n, kwargs.get('plural', {})))
        phrase.append(i == len(count) - 2 and ' and ' or ', ')
    return ''.join(phrase[:-1])
quantify = count
readable_types = (("^<type '", ''), ("^<class '(.*)'\\>", '\\1 class'), ("'>", ''), ('pyobjc', 'PyObjC'), ('objc_class', 'Objective-C class'), ('objc', 'Objective-C'), ('<objective-c class  (.*) at [0-9][0-9|a-z]*>', 'Objective-C \\1 class'), ('bool', 'boolean'), ('int', 'integer'), ('long', 'long integer'), ('float', 'float'), ('str', 'string'), ('unicode', 'string'), ('dict', 'dictionary'), ('NoneType', 'None type'), ('instancemethod', 'instance method'), ('builtin_function_or_method', 'built-in function'), ('classobj', 'class object'), ('\\.', ' '), ('_', ' '))

def reflect(object, quantify=True, replace=readable_types):
    if False:
        i = 10
        return i + 15
    ' Returns the type of each object in the given object.\n        - For modules, this means classes and functions etc.\n        - For list and tuples, means the type of each item in it.\n        - For other objects, means the type of the object itself.\n    '
    _type = lambda object: type(object).__name__
    types = []
    if hasattr(object, '__dict__'):
        if _type(object) in ('function', 'instancemethod'):
            types.append(_type(object))
        else:
            for v in object.__dict__.values():
                try:
                    types.append(str(v.__classname__))
                except:
                    types.append(_type(v))
    elif isinstance(object, (list, tuple, set)):
        types += [_type(x) for x in object]
    elif isinstance(object, dict):
        types += [_type(k) for k in object]
        types += [_type(v) for v in object.values()]
    else:
        types.append(_type(object))
    m = {}
    for i in range(len(types)):
        k = types[i]
        if k not in m:
            for (a, b) in replace:
                types[i] = re.sub(a, b, types[i])
            m[k] = types[i]
        types[i] = m[k]
    if not quantify:
        if not isinstance(object, (list, tuple, set, dict)) and (not hasattr(object, '__dict__')):
            return types[0]
        return types
    return count(types, plural={'built-in function': 'built-in functions'})
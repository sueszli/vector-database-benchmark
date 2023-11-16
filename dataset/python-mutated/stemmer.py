from __future__ import unicode_literals
from __future__ import division
import re
from builtins import str, bytes, dict, int
from builtins import object, range
VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']
DOUBLE = ['bb', 'dd', 'ff', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt']
VALID_LI = ['b', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't']

def is_vowel(s):
    if False:
        print('Hello World!')
    return s in VOWELS

def is_consonant(s):
    if False:
        print('Hello World!')
    return s not in VOWELS

def is_double_consonant(s):
    if False:
        i = 10
        return i + 15
    return s in DOUBLE

def is_short_syllable(w, before=None):
    if False:
        for i in range(10):
            print('nop')
    ' A short syllable in a word is either:\n        - a vowel followed by a non-vowel other than w, x or Y and preceded by a non-vowel\n        - a vowel at the beginning of the word followed by a non-vowel. \n        Checks the three characters before the given index in the word (or entire word if None).\n    '
    if before is not None:
        i = before < 0 and len(w) + before or before
        return is_short_syllable(w[max(0, i - 3):i])
    if len(w) == 3 and is_consonant(w[0]) and is_vowel(w[1]) and is_consonant(w[2]) and (w[2] not in 'wxY'):
        return True
    if len(w) == 2 and is_vowel(w[0]) and is_consonant(w[1]):
        return True
    return False

def is_short(w):
    if False:
        while True:
            i = 10
    ' A word is called short if it consists of a short syllable preceded by zero or more consonants. \n    '
    return is_short_syllable(w[-3:]) and len([ch for ch in w[:-3] if ch in VOWELS]) == 0
overstemmed = ('gener', 'commun', 'arsen')
RE_R1 = re.compile('[aeiouy][^aeiouy]')

def R1(w):
    if False:
        return 10
    ' R1 is the region after the first non-vowel following a vowel, \n        or the end of the word if there is no such non-vowel. \n    '
    m = RE_R1.search(w)
    if m:
        return w[m.end():]
    return ''

def R2(w):
    if False:
        print('Hello World!')
    ' R2 is the region after the first non-vowel following a vowel in R1, \n        or the end of the word if there is no such non-vowel.\n    '
    if w.startswith(tuple(overstemmed)):
        return R1(R1(R1(w)))
    return R1(R1(w))

def find_vowel(w):
    if False:
        for i in range(10):
            print('nop')
    ' Returns the index of the first vowel in the word.\n        When no vowel is found, returns len(word).\n    '
    for (i, ch) in enumerate(w):
        if ch in VOWELS:
            return i
    return len(w)

def has_vowel(w):
    if False:
        i = 10
        return i + 15
    ' Returns True if there is a vowel in the given string.\n    '
    for ch in w:
        if ch in VOWELS:
            return True
    return False

def vowel_consonant_pairs(w, max=None):
    if False:
        return 10
    ' Returns the number of consecutive vowel-consonant pairs in the word.\n    '
    m = 0
    for (i, ch) in enumerate(w):
        if is_vowel(ch) and i < len(w) - 1 and is_consonant(w[i + 1]):
            m += 1
            if m == max:
                break
    return m

def step_1a(w):
    if False:
        print('Hello World!')
    ' Step 1a handles -s suffixes.\n    '
    if w.endswith('s'):
        if w.endswith('sses'):
            return w[:-2]
        if w.endswith('ies'):
            return len(w) == 4 and w[:-1] or w[:-2]
        if w.endswith(('us', 'ss')):
            return w
        if find_vowel(w) < len(w) - 2:
            return w[:-1]
    return w

def step_1b(w):
    if False:
        print('Hello World!')
    ' Step 1b handles -ed and -ing suffixes (or -edly and -ingly).\n        Removes double consonants at the end of the stem and adds -e to some words.\n    '
    if w.endswith('y') and w.endswith(('edly', 'ingly')):
        w = w[:-2]
    if w.endswith(('ed', 'ing')):
        if w.endswith('ied'):
            return len(w) == 4 and w[:-1] or w[:-2]
        if w.endswith('eed'):
            return R1(w).endswith('eed') and w[:-1] or w
        for suffix in ('ed', 'ing'):
            if w.endswith(suffix) and has_vowel(w[:-len(suffix)]):
                w = w[:-len(suffix)]
                if w.endswith(('at', 'bl', 'iz')):
                    return w + 'e'
                if is_double_consonant(w[-2:]):
                    return w[:-1]
                if is_short(w):
                    return w + 'e'
    return w

def step_1c(w):
    if False:
        while True:
            i = 10
    ' Step 1c replaces suffix -y or -Y by -i if preceded by a non-vowel \n        which is not the first letter of the word (cry => cri, by => by, say => say).\n    '
    if len(w) > 2 and w.endswith(('y', 'Y')) and is_consonant(w[-2]):
        return w[:-1] + 'i'
    return w
suffixes2 = [('al', (('ational', 'ate'), ('tional', 'tion'))), ('ci', (('enci', 'ence'), ('anci', 'ance'))), ('er', (('izer', 'ize'),)), ('li', (('bli', 'ble'), ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'))), ('on', (('ization', 'ize'), ('isation', 'ize'), ('ation', 'ate'))), ('or', (('ator', 'ate'),)), ('ss', (('iveness', 'ive'), ('fulness', 'ful'), ('ousness', 'ous'))), ('sm', (('alism', 'al'),)), ('ti', (('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'))), ('gi', (('logi', 'log'),))]

def step_2(w):
    if False:
        return 10
    ' Step 2 replaces double suffixes (singularization => singularize).\n        This only happens if there is at least one vowel-consonant pair before the suffix.\n    '
    for (suffix, rules) in suffixes2:
        if w.endswith(suffix):
            for (A, B) in rules:
                if w.endswith(A):
                    return R1(w).endswith(A) and w[:-len(A)] + B or w
    if w.endswith('li') and R1(w)[-3:-2] in VALID_LI:
        return w[:-2]
    return w
suffixes3 = [('e', (('icate', 'ic'), ('ative', ''), ('alize', 'al'))), ('i', (('iciti', 'ic'),)), ('l', (('ical', 'ic'), ('ful', ''))), ('s', (('ness', ''),))]

def step_3(w):
    if False:
        i = 10
        return i + 15
    ' Step 3 replaces -ic, -ful, -ness etc. suffixes.\n        This only happens if there is at least one vowel-consonant pair before the suffix.\n    '
    for (suffix, rules) in suffixes3:
        if w.endswith(suffix):
            for (A, B) in rules:
                if w.endswith(A):
                    return R1(w).endswith(A) and w[:-len(A)] + B or w
    return w
suffixes4 = [('al', ('al',)), ('ce', ('ance', 'ence')), ('er', ('er',)), ('ic', ('ic',)), ('le', ('able', 'ible')), ('nt', ('ant', 'ement', 'ment', 'ent')), ('e', ('ate', 'ive', 'ize')), (('m', 'i', 's'), ('ism', 'iti', 'ous'))]

def step_4(w):
    if False:
        i = 10
        return i + 15
    ' Step 4 strips -ant, -ent etc. suffixes.\n        This only happens if there is more than one vowel-consonant pair before the suffix.\n    '
    for (suffix, rules) in suffixes4:
        if w.endswith(suffix):
            for A in rules:
                if w.endswith(A):
                    return R2(w).endswith(A) and w[:-len(A)] or w
    if R2(w).endswith('ion') and w[:-3].endswith(('s', 't')):
        return w[:-3]
    return w

def step_5a(w):
    if False:
        for i in range(10):
            print('nop')
    ' Step 5a strips suffix -e if preceded by multiple vowel-consonant pairs,\n        or one vowel-consonant pair that is not a short syllable.\n    '
    if w.endswith('e'):
        if R2(w).endswith('e') or (R1(w).endswith('e') and (not is_short_syllable(w, before=-1))):
            return w[:-1]
    return w

def step_5b(w):
    if False:
        while True:
            i = 10
    ' Step 5b strips suffix -l if preceded by l and multiple vowel-consonant pairs,\n        bell => bell, rebell => rebel.\n    '
    if w.endswith('ll') and R2(w).endswith('l'):
        return w[:-1]
    return w
exceptions = {'skis': 'ski', 'skies': 'sky', 'dying': 'die', 'lying': 'lie', 'tying': 'tie', 'innings': 'inning', 'outings': 'outing', 'cannings': 'canning', 'idly': 'idl', 'gently': 'gentl', 'ugly': 'ugli', 'early': 'earli', 'only': 'onli', 'singly': 'singl'}
uninflected = dict.fromkeys(['sky', 'news', 'howe', 'inning', 'outing', 'canning', 'proceed', 'exceed', 'succeed', 'atlas', 'cosmos', 'bias', 'andes'], True)

def case_sensitive(stem, word):
    if False:
        i = 10
        return i + 15
    ' Applies the letter case of the word to the stem:\n        Ponies => Poni\n    '
    ch = []
    for i in range(len(stem)):
        if word[i] == word[i].upper():
            ch.append(stem[i].upper())
        else:
            ch.append(stem[i])
    return ''.join(ch)

def upper_consonant_y(w):
    if False:
        print('Hello World!')
    ' Sets the initial y, or y after a vowel, to Y.\n        Of course, y is interpreted as a vowel and Y as a consonant.\n    '
    a = []
    p = None
    for ch in w:
        if ch == 'y' and (p is None or p in VOWELS):
            a.append('Y')
        else:
            a.append(ch)
        p = ch
    return ''.join(a)
cache = {}

def stem(word, cached=True, history=10000, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' Returns the stem of the given word: ponies => poni.\n        Note: it is often taken to be a crude error \n        that a stemming algorithm does not leave a real word after removing the stem. \n        But the purpose of stemming is to bring variant forms of a word together, \n        not to map a word onto its "paradigm" form. \n    '
    stem = word.lower()
    if cached and stem in cache:
        return case_sensitive(cache[stem], word)
    if cached and len(cache) > history:
        cache.clear()
    if len(stem) <= 2:
        return case_sensitive(stem, word)
    if stem in exceptions:
        return case_sensitive(exceptions[stem], word)
    if stem in uninflected:
        return case_sensitive(stem, word)
    stem = upper_consonant_y(stem)
    for f in (step_1a, step_1b, step_1c, step_2, step_3, step_4, step_5a, step_5b):
        stem = f(stem)
    stem = stem.lower()
    stem = case_sensitive(stem, word)
    if cached:
        cache[word.lower()] = stem.lower()
    return stem
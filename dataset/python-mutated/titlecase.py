"""
Original Perl version by: John Gruber https://daringfireball.net/ 10 May 2008
Python version by Stuart Colville http://muffinresearch.co.uk
Modifications to make it work with non-ascii chars by Kovid Goyal
License: http://www.opensource.org/licenses/mit-license.php
"""
import re
from calibre.utils.icu import capitalize, lower as icu_lower, upper as icu_upper
__all__ = ['titlecase']
__version__ = '0.5'
SMALL = 'a|an|and|as|at|but|by|en|for|if|in|of|on|or|the|to|v\\.?|via|vs\\.?'
PUNCT = '!"#$%&\'‘’()*+,\\-‒–—―./:;?@[\\\\\\]_`{|}~'
SMALL_WORDS = re.compile('^(%s)$' % SMALL, re.I)
INLINE_PERIOD = re.compile('[a-z][.][a-z]', re.I)
UC_ELSEWHERE = re.compile('[%s]*?[a-zA-Z]+[A-Z]+?' % PUNCT)
CAPFIRST = re.compile(str('^[%s]*?(\\w)' % PUNCT), flags=re.UNICODE)
SMALL_FIRST = re.compile(f'^([{PUNCT}]*)({SMALL})\\b', re.I | re.U)
SMALL_LAST = re.compile(f'\\b({SMALL})[{PUNCT}]?$', re.I | re.U)
SMALL_AFTER_NUM = re.compile('(\\d+\\s+)(a|an|the)\\b', re.I | re.U)
SUBPHRASE = re.compile('([:.;?!][ ])(%s)' % SMALL)
APOS_SECOND = re.compile("^[dol]{1}['‘]{1}[a-z]+$", re.I)
UC_INITIALS = re.compile('^(?:[A-Z]{1}\\.{1}|[A-Z]{1}\\.{1}[A-Z]{1})+$')
_lang = None

def lang():
    if False:
        return 10
    global _lang
    if _lang is None:
        from calibre.utils.localization import get_lang
        _lang = get_lang().lower()
    return _lang

def titlecase(text):
    if False:
        return 10
    '\n    Titlecases input text\n\n    This filter changes all words to Title Caps, and attempts to be clever\n    about *un*capitalizing SMALL words like a/an/the in the input.\n\n    The list of "SMALL words" which are not capped comes from\n    the New York Times Manual of Style, plus \'vs\' and \'v\'.\n\n    '
    all_caps = icu_upper(text) == text
    pat = re.compile('(\\s+)')
    line = []
    for word in pat.split(text):
        if not word:
            continue
        if pat.match(word) is not None:
            line.append(word)
            continue
        if all_caps:
            if UC_INITIALS.match(word):
                line.append(word)
                continue
            else:
                word = icu_lower(word)
        if APOS_SECOND.match(word):
            word = word.replace(word[0], icu_upper(word[0]), 1)
            word = word[:2] + icu_upper(word[2]) + word[3:]
            line.append(word)
            continue
        if INLINE_PERIOD.search(word) or UC_ELSEWHERE.match(word):
            line.append(word)
            continue
        if SMALL_WORDS.match(word):
            line.append(icu_lower(word))
            continue
        hyphenated = []
        for item in word.split('-'):
            hyphenated.append(CAPFIRST.sub(lambda m: icu_upper(m.group(0)), item))
        line.append('-'.join(hyphenated))
    result = ''.join(line)
    result = SMALL_FIRST.sub(lambda m: '{}{}'.format(m.group(1), capitalize(m.group(2))), result)
    result = SMALL_AFTER_NUM.sub(lambda m: '{}{}'.format(m.group(1), capitalize(m.group(2))), result)
    result = SMALL_LAST.sub(lambda m: capitalize(m.group(0)), result)
    result = SUBPHRASE.sub(lambda m: '{}{}'.format(m.group(1), capitalize(m.group(2))), result)
    return result
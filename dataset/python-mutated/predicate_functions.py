import re
from itertools import chain
from math import copysign, floor, log10
from typing import Any, FrozenSet, Sequence, Tuple, Union
from doublemetaphone import doublemetaphone
from dedupe.cpredicates import initials, ngrams, unique_ngrams
words = re.compile("[\\w']+").findall
integers = re.compile('\\d+').findall
start_word = re.compile("^([\\w']+)").match
two_start_words = re.compile("^([\\w']+\\W+[\\w']+)").match
start_integer = re.compile('^(\\d+)').match
alpha_numeric = re.compile('(?=[a-zA-Z]*\\d)[a-zA-Z\\d]+').findall

def wholeFieldPredicate(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    'return the whole field as a string'
    return frozenset((str(field),))

def tokenFieldPredicate(field: str) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    'returns the tokens'
    return frozenset(words(field))

def firstTokenPredicate(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    first_token = start_word(field)
    if first_token:
        return frozenset(first_token.groups())
    else:
        return frozenset()

def firstTwoTokensPredicate(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    first_two_tokens = two_start_words(field)
    if first_two_tokens:
        return frozenset(first_two_tokens.groups())
    else:
        return frozenset()

def commonIntegerPredicate(field: str) -> FrozenSet[str]:
    if False:
        return 10
    'return any integers'
    return frozenset((str(int(i)) for i in integers(field)))

def alphaNumericPredicate(field: str) -> FrozenSet[str]:
    if False:
        return 10
    return frozenset(alpha_numeric(field))

def nearIntegersPredicate(field: str) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    'for any integer N in field return the integers N-1, N and N+1'
    string_ints = integers(field)
    near_ints = set()
    for s in string_ints:
        num = int(s)
        near_ints.add(str(num - 1))
        near_ints.add(str(num))
        near_ints.add(str(num + 1))
    return frozenset(near_ints)

def hundredIntegerPredicate(field: str) -> FrozenSet[str]:
    if False:
        return 10
    return frozenset((str(int(i))[:-2] + '00' for i in integers(field)))

def hundredIntegersOddPredicate(field: str) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    return frozenset((str(int(i))[:-2] + '0' + str(int(i) % 2) for i in integers(field)))

def firstIntegerPredicate(field: str) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    first_token = start_integer(field)
    if first_token:
        return frozenset(first_token.groups())
    else:
        return frozenset()

def ngramsTokens(field: Sequence[Any], n: int) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    grams = set()
    n_tokens = len(field)
    for i in range(n_tokens):
        for j in range(i + n, min(n_tokens, i + n) + 1):
            grams.add(' '.join((str(tok) for tok in field[i:j])))
    return frozenset(grams)

def commonTwoTokens(field: str) -> FrozenSet[str]:
    if False:
        return 10
    return ngramsTokens(field.split(), 2)

def commonThreeTokens(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    return ngramsTokens(field.split(), 3)

def fingerprint(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    return frozenset((''.join(sorted(field.split())),))

def oneGramFingerprint(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    return frozenset((''.join(sorted({*field.replace(' ', '')})),))

def twoGramFingerprint(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    if len(field) > 1:
        return frozenset((''.join(sorted(unique_ngrams(field.replace(' ', ''), 2))),))
    else:
        return frozenset()

def commonFourGram(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    'return 4-grams'
    return frozenset(unique_ngrams(field.replace(' ', ''), 4))

def commonSixGram(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    'return 6-grams'
    return frozenset(unique_ngrams(field.replace(' ', ''), 6))

def sameThreeCharStartPredicate(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    'return first three characters'
    return frozenset(initials(field.replace(' ', ''), 3))

def sameFiveCharStartPredicate(field: str) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    'return first five characters'
    return frozenset(initials(field.replace(' ', ''), 5))

def sameSevenCharStartPredicate(field: str) -> FrozenSet[str]:
    if False:
        i = 10
        return i + 15
    'return first seven characters'
    return frozenset(initials(field.replace(' ', ''), 7))

def suffixArray(field: str) -> FrozenSet[str]:
    if False:
        i = 10
        return i + 15
    n = len(field) - 4
    if n > 0:
        return frozenset((field[i:] for i in range(0, n)))
    else:
        return frozenset()

def sortedAcronym(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    return frozenset((''.join(sorted((each[0] for each in field.split()))),))

def doubleMetaphone(field: str) -> FrozenSet[str]:
    if False:
        return 10
    return frozenset((metaphone for metaphone in doublemetaphone(field) if metaphone))

def metaphoneToken(field: str) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    return frozenset((metaphone_token for metaphone_token in chain(*(doublemetaphone(token) for token in field.split())) if metaphone_token))

def wholeSetPredicate(field_set: Sequence[Any]) -> FrozenSet[str]:
    if False:
        i = 10
        return i + 15
    return frozenset((str(field_set),))

def commonSetElementPredicate(field_set: Sequence[Any]) -> FrozenSet[str]:
    if False:
        i = 10
        return i + 15
    'return set as individual elements'
    return frozenset((str(item) for item in field_set))

def commonTwoElementsPredicate(field: Sequence[Any]) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    return ngramsTokens(sorted(field), 2)

def commonThreeElementsPredicate(field: Sequence[Any]) -> FrozenSet[str]:
    if False:
        return 10
    return ngramsTokens(sorted(field), 3)

def lastSetElementPredicate(field_set: Sequence[Any]) -> FrozenSet[str]:
    if False:
        return 10
    return frozenset((str(max(field_set)),))

def firstSetElementPredicate(field_set: Sequence[Any]) -> FrozenSet[str]:
    if False:
        while True:
            i = 10
    return frozenset((str(min(field_set)),))

def magnitudeOfCardinality(field_set: Sequence[Any]) -> FrozenSet[str]:
    if False:
        print('Hello World!')
    return orderOfMagnitude(len(field_set))

def latLongGridPredicate(field: Tuple[float], digits: int=1) -> FrozenSet[str]:
    if False:
        return 10
    '\n    Given a lat / long pair, return the grid coordinates at the\n    nearest base value.  e.g., (42.3, -5.4) returns a grid at 0.1\n    degree resolution of 0.1 degrees of latitude ~ 7km, so this is\n    effectively a 14km lat grid.  This is imprecise for longitude,\n    since 1 degree of longitude is 0km at the poles, and up to 111km\n    at the equator. But it should be reasonably precise given some\n    prior logical block (e.g., country).\n    '
    if any(field):
        return frozenset((str(tuple((round(dim, digits) for dim in field))),))
    else:
        return frozenset()

def orderOfMagnitude(field: Union[int, float]) -> FrozenSet[str]:
    if False:
        for i in range(10):
            print('nop')
    if field > 0:
        return frozenset((str(int(round(log10(field)))),))
    else:
        return frozenset()

def roundTo1(field: float) -> FrozenSet[str]:
    if False:
        i = 10
        return i + 15
    abs_num = abs(field)
    order = int(floor(log10(abs_num)))
    rounded = round(abs_num, -order)
    return frozenset((str(int(copysign(rounded, field))),))
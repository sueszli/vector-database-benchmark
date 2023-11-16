""" from https://github.com/keithito/tacotron """
import re
from typing import Dict
import inflect
_inflect = inflect.engine()
_comma_number_re = re.compile('([0-9][0-9\\,]+[0-9])')
_decimal_number_re = re.compile('([0-9]+\\.[0-9]+)')
_currency_re = re.compile('(£|\\$|¥)([0-9\\,\\.]*[0-9]+)')
_ordinal_re = re.compile('[0-9]+(st|nd|rd|th)')
_number_re = re.compile('-?[0-9]+')

def _remove_commas(m):
    if False:
        return 10
    return m.group(1).replace(',', '')

def _expand_decimal_point(m):
    if False:
        for i in range(10):
            print('nop')
    return m.group(1).replace('.', ' point ')

def __expand_currency(value: str, inflection: Dict[float, str]) -> str:
    if False:
        return 10
    parts = value.replace(',', '').split('.')
    if len(parts) > 2:
        return f'{value} {inflection[2]}'
    text = []
    integer = int(parts[0]) if parts[0] else 0
    if integer > 0:
        integer_unit = inflection.get(integer, inflection[2])
        text.append(f'{integer} {integer_unit}')
    fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if fraction > 0:
        fraction_unit = inflection.get(fraction / 100, inflection[0.02])
        text.append(f'{fraction} {fraction_unit}')
    if len(text) == 0:
        return f'zero {inflection[2]}'
    return ' '.join(text)

def _expand_currency(m: 're.Match') -> str:
    if False:
        return 10
    currencies = {'$': {0.01: 'cent', 0.02: 'cents', 1: 'dollar', 2: 'dollars'}, '€': {0.01: 'cent', 0.02: 'cents', 1: 'euro', 2: 'euros'}, '£': {0.01: 'penny', 0.02: 'pence', 1: 'pound sterling', 2: 'pounds sterling'}, '¥': {0.02: 'sen', 2: 'yen'}}
    unit = m.group(1)
    currency = currencies[unit]
    value = m.group(2)
    return __expand_currency(value, currency)

def _expand_ordinal(m):
    if False:
        print('Hello World!')
    return _inflect.number_to_words(m.group(0))

def _expand_number(m):
    if False:
        return 10
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return 'two thousand'
        if 2000 < num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        if num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    return _inflect.number_to_words(num, andword='')

def normalize_numbers(text):
    if False:
        print('Hello World!')
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
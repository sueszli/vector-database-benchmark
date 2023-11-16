"""
Formatting numeric literals.
"""
from blib2to3.pytree import Leaf

def format_hex(text: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Formats a hexadecimal string like "0x12B3"\n    '
    (before, after) = (text[:2], text[2:])
    return f'{before}{after.upper()}'

def format_scientific_notation(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Formats a numeric string utilizing scentific notation'
    (before, after) = text.split('e')
    sign = ''
    if after.startswith('-'):
        after = after[1:]
        sign = '-'
    elif after.startswith('+'):
        after = after[1:]
    before = format_float_or_int_string(before)
    return f'{before}e{sign}{after}'

def format_complex_number(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Formats a complex string like `10j`'
    number = text[:-1]
    suffix = text[-1]
    return f'{format_float_or_int_string(number)}{suffix}'

def format_float_or_int_string(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Formats a float string like "1.0".'
    if '.' not in text:
        return text
    (before, after) = text.split('.')
    return f'{before or 0}.{after or 0}'

def normalize_numeric_literal(leaf: Leaf) -> None:
    if False:
        i = 10
        return i + 15
    'Normalizes numeric (float, int, and complex) literals.\n\n    All letters used in the representation are normalized to lowercase.'
    text = leaf.value.lower()
    if text.startswith(('0o', '0b')):
        pass
    elif text.startswith('0x'):
        text = format_hex(text)
    elif 'e' in text:
        text = format_scientific_notation(text)
    elif text.endswith('j'):
        text = format_complex_number(text)
    else:
        text = format_float_or_int_string(text)
    leaf.value = text
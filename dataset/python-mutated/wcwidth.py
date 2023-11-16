import unicodedata
from functools import lru_cache

@lru_cache(100)
def wcwidth(c: str) -> int:
    if False:
        return 10
    'Determine how many columns are needed to display a character in a terminal.\n\n    Returns -1 if the character is not printable.\n    Returns 0, 1 or 2 for other characters.\n    '
    o = ord(c)
    if 32 <= o < 127:
        return 1
    if o == 0 or 8203 <= o <= 8207 or 8232 <= o <= 8238 or (8288 <= o <= 8291):
        return 0
    category = unicodedata.category(c)
    if category == 'Cc':
        return -1
    if category in ('Me', 'Mn'):
        return 0
    if unicodedata.east_asian_width(c) in ('F', 'W'):
        return 2
    return 1

def wcswidth(s: str) -> int:
    if False:
        while True:
            i = 10
    'Determine how many columns are needed to display a string in a terminal.\n\n    Returns -1 if the string contains non-printable characters.\n    '
    width = 0
    for c in unicodedata.normalize('NFC', s):
        wc = wcwidth(c)
        if wc < 0:
            return -1
        width += wc
    return width
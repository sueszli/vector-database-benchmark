"""
This module offers a generic Easter computing method for any given year, using
Western, Orthodox or Julian algorithms.
"""
import datetime
__all__ = ['easter', 'EASTER_JULIAN', 'EASTER_ORTHODOX', 'EASTER_WESTERN']
EASTER_JULIAN = 1
EASTER_ORTHODOX = 2
EASTER_WESTERN = 3

def easter(year, method=EASTER_WESTERN):
    if False:
        for i in range(10):
            print('nop')
    '\n    This method was ported from the work done by GM Arts,\n    on top of the algorithm by Claus Tondering, which was\n    based in part on the algorithm of Ouding (1940), as\n    quoted in "Explanatory Supplement to the Astronomical\n    Almanac", P.  Kenneth Seidelmann, editor.\n\n    This algorithm implements three different Easter\n    calculation methods:\n\n    1. Original calculation in Julian calendar, valid in\n       dates after 326 AD\n    2. Original method, with date converted to Gregorian\n       calendar, valid in years 1583 to 4099\n    3. Revised method, in Gregorian calendar, valid in\n       years 1583 to 4099 as well\n\n    These methods are represented by the constants:\n\n    * ``EASTER_JULIAN   = 1``\n    * ``EASTER_ORTHODOX = 2``\n    * ``EASTER_WESTERN  = 3``\n\n    The default method is method 3.\n\n    More about the algorithm may be found at:\n\n    `GM Arts: Easter Algorithms <http://www.gmarts.org/index.php?go=415>`_\n\n    and\n\n    `The Calendar FAQ: Easter <https://www.tondering.dk/claus/cal/easter.php>`_\n\n    '
    if not 1 <= method <= 3:
        raise ValueError('invalid method')
    y = year
    g = y % 19
    e = 0
    if method < 3:
        i = (19 * g + 15) % 30
        j = (y + y // 4 + i) % 7
        if method == 2:
            e = 10
            if y > 1600:
                e = e + y // 100 - 16 - (y // 100 - 16) // 4
    else:
        c = y // 100
        h = (c - c // 4 - (8 * c + 13) // 25 + 19 * g + 15) % 30
        i = h - h // 28 * (1 - h // 28 * (29 // (h + 1)) * ((21 - g) // 11))
        j = (y + y // 4 + i + 2 - c + c // 4) % 7
    p = i - j + e
    d = 1 + (p + 27 + (p + 6) // 40) % 31
    m = 3 + (p + 26) // 30
    return datetime.date(int(y), int(m), int(d))
"""Functions for natural sorting of strings containing numbers.
"""
import re
from picard.util import strxfrm
RE_NUMBER = re.compile('(\\d+)')

def natkey(text):
    if False:
        while True:
            i = 10
    '\n    Return a sort key for a string for natural sort order.\n    '
    return [int(s) if s.isdecimal() else strxfrm(s) for s in RE_NUMBER.split(str(text).replace('\x00', ''))]

def natsorted(values):
    if False:
        return 10
    "\n    Returns a copy of the given list sorted naturally.\n\n    >>> sort(['track02', 'track10', 'track1'])\n    ['track1', 'track02', 'track10']\n    "
    return sorted(values, key=natkey)
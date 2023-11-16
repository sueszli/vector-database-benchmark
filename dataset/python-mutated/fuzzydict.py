"""Match items in a dictionary using fuzzy matching

Implemented for pywinauto.

This class uses difflib to match strings.
This class uses a linear search to find the items as it HAS to iterate over
every item in the dictionary (otherwise it would not be possible to know which
is the 'best' match).

If the exact item is in the dictionary (no fuzzy matching needed - then it
doesn't do the linear search and speed should be similar to standard Python
dictionaries.

>>> fuzzywuzzy = FuzzyDict({"hello" : "World", "Hiya" : 2, "Here you are" : 3})
>>> fuzzywuzzy['Me again'] = [1,2,3]
>>>
>>> fuzzywuzzy['Hi']
2
>>>
>>>
>>> # next one doesn't match well enough - so a key error is raised
...
>>> fuzzywuzzy['There']
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
  File "pywinauto\x0cuzzydict.py", line 125, in __getitem__
    raise KeyError(
KeyError: "'There'. closest match: 'hello' with ratio 0.400"
>>>
>>> fuzzywuzzy['you are']
3
>>> fuzzywuzzy['again']
[1, 2, 3]
>>>
"""
from __future__ import unicode_literals
import difflib
from collections import OrderedDict

class FuzzyDict(OrderedDict):
    """Provides a dictionary that performs fuzzy lookup"""

    def __init__(self, items=None, cutoff=0.6):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new FuzzyDict instance\n\n        items is an dictionary to copy items from (optional)\n        cutoff is the match ratio below which mathes should not be considered\n        cutoff needs to be a float between 0 and 1 (where zero is no match\n        and 1 is a perfect match).\n        '
        super(FuzzyDict, self).__init__()
        self._dict_contains = lambda key: super(FuzzyDict, self).__contains__(key)
        self._dict_getitem = lambda key: super(FuzzyDict, self).__getitem__(key)
        self.cutoff = cutoff
        if items:
            self.update(items)

    def _search(self, lookfor, stop_on_first=False):
        if False:
            while True:
                i = 10
        '\n        Returns the value whose key best matches lookfor\n\n        if stop_on_first is True then the method returns as soon\n        as it finds the first item.\n        '
        if self._dict_contains(lookfor):
            return (True, lookfor, self._dict_getitem(lookfor), 1)
        ratio_calc = difflib.SequenceMatcher()
        ratio_calc.set_seq1(lookfor)
        best_ratio = 0
        best_match = None
        best_key = None
        for key in self:
            try:
                ratio_calc.set_seq2(key)
            except TypeError:
                continue
            try:
                ratio = ratio_calc.ratio()
            except TypeError:
                break
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
                best_match = self._dict_getitem(key)
            if stop_on_first and ratio >= self.cutoff:
                break
        return (best_ratio >= self.cutoff, best_key, best_match, best_ratio)

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Overides OrderedDict __contains__ to use fuzzy matching'
        if self._search(item, True)[0]:
            return True
        else:
            return False

    def __getitem__(self, lookfor):
        if False:
            i = 10
            return i + 15
        'Overides OrderedDict __getitem__ to use fuzzy matching'
        (matched, key, item, ratio) = self._search(lookfor)
        if not matched:
            raise KeyError("'{0}'. closest match: '{1}' with ratio {2}".format(str(lookfor), str(key), ratio))
        return item
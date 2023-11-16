import os
import re
from difflib import SequenceMatcher
from autojump_utils import is_python3
from autojump_utils import last
if is_python3():
    ifilter = filter
    imap = map
    os.getcwdu = os.getcwd
else:
    from itertools import ifilter
    from itertools import imap

def match_anywhere(needles, haystack, ignore_case=False):
    if False:
        while True:
            i = 10
    "\n    Matches needles anywhere in the path as long as they're in the same (but\n    not necessary consecutive) order.\n\n    For example:\n        needles = ['foo', 'baz']\n        regex needle = r'.*foo.*baz.*'\n        haystack = [\n            (path='/foo/bar/baz', weight=10),\n            (path='/baz/foo/bar', weight=10),\n            (path='/foo/baz', weight=10),\n        ]\n\n        result = [\n            (path='/moo/foo/baz', weight=10),\n            (path='/foo/baz', weight=10),\n        ]\n    "
    regex_needle = '.*' + '.*'.join(imap(re.escape, needles)) + '.*'
    regex_flags = re.IGNORECASE | re.UNICODE if ignore_case else re.UNICODE
    found = lambda haystack: re.search(regex_needle, haystack.path, flags=regex_flags)
    return ifilter(found, haystack)

def match_consecutive(needles, haystack, ignore_case=False):
    if False:
        print('Hello World!')
    "\n    Matches consecutive needles at the end of a path.\n\n    For example:\n        needles = ['foo', 'baz']\n        haystack = [\n            (path='/foo/bar/baz', weight=10),\n            (path='/foo/baz/moo', weight=10),\n            (path='/moo/foo/baz', weight=10),\n            (path='/foo/baz', weight=10),\n        ]\n\n        # We can't actually use re.compile because of re.UNICODE\n        regex_needle = re.compile(r'''\n            foo     # needle #1\n            [^/]*   # all characters except os.sep zero or more times\n            /       # os.sep\n            [^/]*   # all characters except os.sep zero or more times\n            baz     # needle #2\n            [^/]*   # all characters except os.sep zero or more times\n            $       # end of string\n            ''')\n\n        result = [\n            (path='/moo/foo/baz', weight=10),\n            (path='/foo/baz', weight=10),\n        ]\n    "
    regex_no_sep = '[^' + os.sep + ']*'
    regex_no_sep_end = regex_no_sep + '$'
    regex_one_sep = regex_no_sep + os.sep + regex_no_sep
    regex_needle = regex_one_sep.join(imap(re.escape, needles)) + regex_no_sep_end
    regex_flags = re.IGNORECASE | re.UNICODE if ignore_case else re.UNICODE
    found = lambda entry: re.search(regex_needle, entry.path, flags=regex_flags)
    return ifilter(found, haystack)

def match_fuzzy(needles, haystack, ignore_case=False, threshold=0.6):
    if False:
        print('Hello World!')
    "\n    Performs an approximate match with the last needle against the end of\n    every path past an acceptable threshold.\n\n    For example:\n        needles = ['foo', 'bar']\n        haystack = [\n            (path='/foo/bar/baz', weight=11),\n            (path='/foo/baz/moo', weight=10),\n            (path='/moo/foo/baz', weight=10),\n            (path='/foo/baz', weight=10),\n            (path='/foo/bar', weight=10),\n        ]\n\n    result = [\n            (path='/foo/bar/baz', weight=11),\n            (path='/moo/foo/baz', weight=10),\n            (path='/foo/baz', weight=10),\n            (path='/foo/bar', weight=10),\n        ]\n\n    This is a weak heuristic and used as a last resort to find matches.\n    "
    end_dir = lambda path: last(os.path.split(path))
    if ignore_case:
        needle = last(needles).lower()
        match_percent = lambda entry: SequenceMatcher(a=needle, b=end_dir(entry.path.lower())).ratio()
    else:
        needle = last(needles)
        match_percent = lambda entry: SequenceMatcher(a=needle, b=end_dir(entry.path)).ratio()
    meets_threshold = lambda entry: match_percent(entry) >= threshold
    return ifilter(meets_threshold, haystack)
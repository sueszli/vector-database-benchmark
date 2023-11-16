"""
Functions to load the test cases ("koans") that make up the
Path to Enlightenment.
"""
import io
import unittest
KOANS_FILENAME = 'koans.txt'

def filter_koan_names(lines):
    if False:
        i = 10
        return i + 15
    '\n    Strips leading and trailing whitespace, then filters out blank\n    lines and comment lines.\n    '
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        if line:
            yield line
    return

def names_from_file(filename):
    if False:
        i = 10
        return i + 15
    '\n    Opens the given ``filename`` and yields the fully-qualified names\n    of TestCases found inside (one per line).\n    '
    with io.open(filename, 'rt', encoding='utf8') as names_file:
        for name in filter_koan_names(names_file):
            yield name
    return

def koans_suite(names):
    if False:
        print('Hello World!')
    '\n    Returns a ``TestSuite`` loaded with all tests found in the given\n    ``names``, preserving the order in which they are found.\n    '
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    for name in names:
        tests = loader.loadTestsFromName(name)
        suite.addTests(tests)
    return suite

def koans(filename=KOANS_FILENAME):
    if False:
        print('Hello World!')
    '\n    Returns a ``TestSuite`` loaded with all the koans (``TestCase``s)\n    listed in ``filename``.\n    '
    names = names_from_file(filename)
    return koans_suite(names)
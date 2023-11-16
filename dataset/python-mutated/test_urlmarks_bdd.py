import os.path
import pytest
import pytest_bdd as bdd
from helpers import testutils
bdd.scenarios('urlmarks.feature')

@pytest.fixture(autouse=True)
def clear_marks(quteproc):
    if False:
        while True:
            i = 10
    'Clear all existing marks between tests.'
    yield
    quteproc.send_cmd(':quickmark-del --all')
    quteproc.wait_for(message='Quickmarks cleared.')
    quteproc.send_cmd(':bookmark-del --all')
    quteproc.wait_for(message='Bookmarks cleared.')

def _check_marks(quteproc, quickmarks, expected, contains):
    if False:
        return 10
    'Make sure the given line does (not) exist in the bookmarks.\n\n    Args:\n        quickmarks: True to check the quickmarks file instead of bookmarks.\n        expected: The line to search for.\n        contains: True if the line should be there, False otherwise.\n    '
    if quickmarks:
        mark_file = os.path.join(quteproc.basedir, 'config', 'quickmarks')
    else:
        mark_file = os.path.join(quteproc.basedir, 'config', 'bookmarks', 'urls')
    quteproc.clear_data()
    quteproc.send_cmd(':save')
    quteproc.wait_for(message='Saved to {}'.format(mark_file))
    with open(mark_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    matched_line = any((testutils.pattern_match(pattern=expected, value=line.rstrip('\n')) for line in lines))
    assert matched_line == contains, lines

@bdd.then(bdd.parsers.parse('the bookmark file should contain "{line}"'))
def bookmark_file_contains(quteproc, line):
    if False:
        i = 10
        return i + 15
    _check_marks(quteproc, quickmarks=False, expected=line, contains=True)

@bdd.then(bdd.parsers.parse('the bookmark file should not contain "{line}"'))
def bookmark_file_does_not_contain(quteproc, line):
    if False:
        i = 10
        return i + 15
    _check_marks(quteproc, quickmarks=False, expected=line, contains=False)

@bdd.then(bdd.parsers.parse('the quickmark file should contain "{line}"'))
def quickmark_file_contains(quteproc, line):
    if False:
        i = 10
        return i + 15
    _check_marks(quteproc, quickmarks=True, expected=line, contains=True)

@bdd.then(bdd.parsers.parse('the quickmark file should not contain "{line}"'))
def quickmark_file_does_not_contain(quteproc, line):
    if False:
        while True:
            i = 10
    _check_marks(quteproc, quickmarks=True, expected=line, contains=False)
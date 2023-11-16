from __future__ import annotations
import pytest
from ansible.utils.vars import isidentifier

@pytest.mark.parametrize('identifier', ['foo', 'foo1_23'])
def test_valid_identifier(identifier):
    if False:
        return 10
    assert isidentifier(identifier)

@pytest.mark.parametrize('identifier', ['pass', 'foo ', ' foo', '1234', '1234abc', '', '   ', 'foo bar', 'no-dashed-names-for-you'])
def test_invalid_identifier(identifier):
    if False:
        print('Hello World!')
    assert not isidentifier(identifier)

def test_keywords_not_in_PY2():
    if False:
        print('Hello World!')
    'In Python 2 ("True", "False", "None") are not keywords. The isidentifier\n    method ensures that those are treated as keywords on both Python 2 and 3.\n    '
    assert not isidentifier('True')
    assert not isidentifier('False')
    assert not isidentifier('None')

def test_non_ascii():
    if False:
        while True:
            i = 10
    'In Python 3 non-ascii characters are allowed as opposed to Python 2. The\n    isidentifier method ensures that those are treated as keywords on both\n    Python 2 and 3.\n    '
    assert not isidentifier('křížek')
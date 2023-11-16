from __future__ import annotations
from ansible.parsing.quoting import unquote
import pytest
UNQUOTE_DATA = ((u'1', u'1'), (u"'1'", u'1'), (u'"1"', u'1'), (u'"1 \'2\'"', u"1 '2'"), (u'\'1 "2"\'', u'1 "2"'), (u"'1 '2''", u"1 '2'"), (u'"1\\"', u'"1\\"'), (u"'1\\'", u"'1\\'"), (u'"1 \\"2\\" 3"', u'1 \\"2\\" 3'), (u"'1 \\'2\\' 3'", u"1 \\'2\\' 3"), (u'"', u'"'), (u"'", u"'"), (u'"1""2"', u'1""2'), (u"'1''2'", u"1''2"), (u'"1" 2 "3"', u'1" 2 "3'), (u'"1"\'2\'"3"', u'1"\'2\'"3'))

@pytest.mark.parametrize('quoted, expected', UNQUOTE_DATA)
def test_unquote(quoted, expected):
    if False:
        print('Hello World!')
    assert unquote(quoted) == expected
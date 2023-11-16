"""String Pattern Matching.

@see: https://docs.python.org/3/tutorial/stdlib.html#string-pattern-matching

The re module provides regular expression tools for advanced string processing.
For complex matching and manipulation, regular expressions offer succinct, optimized solutions:
"""
import re

def test_re():
    if False:
        return 10
    'String Pattern Matching'
    assert re.findall('\\bf[a-z]*', 'which foot or hand fell fastest') == ['foot', 'fell', 'fastest']
    assert re.sub('(\\b[a-z]+) \\1', '\\1', 'cat in the the hat') == 'cat in the hat'
    assert 'tea for too'.replace('too', 'two') == 'tea for two'
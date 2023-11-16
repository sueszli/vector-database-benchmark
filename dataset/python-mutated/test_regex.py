import re
import string
import sys
from functools import reduce
import pytest
from hypothesis import assume, given, reject, strategies as st
from hypothesis.strategies._internal.regex import base_regex_strategy

@st.composite
def charset(draw):
    if False:
        return 10
    negated = draw(st.booleans())
    chars = draw(st.text(string.ascii_letters + string.digits, min_size=1))
    if negated:
        return f'[^{chars}]'
    else:
        return f'[{chars}]'
COMBINED_MATCHER = re.compile('[?+*]{2}')

@st.composite
def conservative_regex(draw):
    if False:
        for i in range(10):
            print('nop')
    result = draw(st.one_of(st.just('.'), st.sampled_from([re.escape(c) for c in string.printable]), charset(), CONSERVATIVE_REGEX.map(lambda s: f'({s})'), CONSERVATIVE_REGEX.map(lambda s: s + '+'), CONSERVATIVE_REGEX.map(lambda s: s + '?'), CONSERVATIVE_REGEX.map(lambda s: s + '*'), st.lists(CONSERVATIVE_REGEX, min_size=1, max_size=3).map('|'.join), st.lists(CONSERVATIVE_REGEX, min_size=1, max_size=3).map(''.join)))
    assume(COMBINED_MATCHER.search(result) is None)
    control = sum((result.count(c) for c in '?+*'))
    assume(control <= 3)
    assume(I_WITH_DOT not in result)
    return result
CONSERVATIVE_REGEX = conservative_regex()
FLAGS = st.sets(st.sampled_from([re.ASCII, re.IGNORECASE, re.MULTILINE, re.DOTALL])).map(lambda flag_set: reduce(int.__or__, flag_set, 0))

@given(st.data())
def test_conservative_regex_are_correct_by_construction(data):
    if False:
        i = 10
        return i + 15
    pattern = re.compile(data.draw(CONSERVATIVE_REGEX), flags=data.draw(FLAGS))
    result = data.draw(base_regex_strategy(pattern, alphabet=st.characters()))
    assume({'ı', 'İ'}.isdisjoint(pattern.pattern + result))
    assert pattern.search(result) is not None

@given(st.data())
def test_fuzz_stuff(data):
    if False:
        for i in range(10):
            print('nop')
    pattern = data.draw(st.text(min_size=1, max_size=5) | st.binary(min_size=1, max_size=5) | CONSERVATIVE_REGEX.filter(bool))
    flags = data.draw(FLAGS)
    try:
        regex = re.compile(pattern, flags=flags)
    except (re.error, FutureWarning):
        reject()
    ex = data.draw(st.from_regex(regex))
    assert regex.search(ex)

@pytest.mark.skipif(sys.version_info[:2] < (3, 11), reason='new syntax')
@given(st.data())
def test_regex_atomic_group(data):
    if False:
        i = 10
        return i + 15
    pattern = 'a(?>bc|b)c'
    ex = data.draw(st.from_regex(pattern))
    assert re.search(pattern, ex)

@pytest.mark.skipif(sys.version_info[:2] < (3, 11), reason='new syntax')
@given(st.data())
def test_regex_possessive(data):
    if False:
        while True:
            i = 10
    pattern = '"[^"]*+"'
    ex = data.draw(st.from_regex(pattern))
    assert re.search(pattern, ex)
I_WITH_DOT = 'İ'
assert I_WITH_DOT.swapcase() == 'i̇'
assert re.compile(I_WITH_DOT, flags=re.IGNORECASE).match(I_WITH_DOT.swapcase())

@given(st.data())
def test_case_insensitive_not_literal_never_constructs_multichar_match(data):
    if False:
        while True:
            i = 10
    pattern = re.compile(f'[^{I_WITH_DOT}]+', flags=re.IGNORECASE)
    strategy = st.from_regex(pattern, fullmatch=True)
    for _ in range(5):
        s = data.draw(strategy)
        assert pattern.fullmatch(s) is not None
        assert set(s).isdisjoint(I_WITH_DOT.swapcase())

@given(st.from_regex(re.compile(f'[^{I_WITH_DOT}_]', re.IGNORECASE), fullmatch=True))
def test_no_error_converting_negated_sets_to_strategy(s):
    if False:
        return 10
    pass
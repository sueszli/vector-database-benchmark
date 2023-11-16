"""This example demonstrates testing a run length encoding scheme. That is, we
take a sequence and represent it by a shorter sequence where each 'run' of
consecutive equal elements is represented as a single element plus a count. So
e.g.

[1, 1, 1, 1, 2, 1] is represented as [[1, 4], [2, 1], [1, 1]]

This demonstrates the useful decode(encode(x)) == x invariant that is often
a fruitful source of testing with Hypothesis.

It also has an example of testing invariants in response to changes in the
underlying data.
"""
from hypothesis import assume, given, strategies as st

def run_length_encode(seq):
    if False:
        i = 10
        return i + 15
    'Encode a sequence as a new run-length encoded sequence.'
    if not seq:
        return []
    result = [[seq[0], 0]]
    for s in seq:
        if s == result[-1][0]:
            result[-1][1] += 1
        else:
            result.append([s, 1])
    return result

def run_length_decode(seq):
    if False:
        print('Hello World!')
    'Take a previously encoded sequence and reconstruct the original from\n    it.'
    result = []
    for (s, i) in seq:
        for _ in range(i):
            result.append(s)
    return result
Lists = st.lists(st.integers(0, 10))

@given(Lists)
def test_decodes_to_starting_sequence(ls):
    if False:
        i = 10
        return i + 15
    "If we encode a sequence and then decode the result, we should get the\n    original sequence back.\n\n    Otherwise we've done something very wrong.\n    "
    assert run_length_decode(run_length_encode(ls)) == ls

@given(Lists, st.data())
def test_duplicating_an_element_does_not_increase_length(ls, data):
    if False:
        for i in range(10):
            print('nop')
    'The previous test could be passed by simply returning the input sequence\n    so we need something that tests the compression property of our encoding.\n\n    In this test we deliberately introduce or extend a run and assert\n    that this does not increase the length of our encoding, because they\n    should be part of the same run in the final result.\n    '
    assume(ls)
    i = data.draw(st.integers(0, len(ls) - 1))
    ls2 = list(ls)
    ls2.insert(i, ls2[i])
    assert len(run_length_encode(ls2)) == len(run_length_encode(ls))
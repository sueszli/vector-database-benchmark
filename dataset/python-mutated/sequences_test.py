import pytest
from allennlp.common.sequences import ConcatenatedSequence

def assert_equal_including_exceptions(expected_fn, actual_fn):
    if False:
        for i in range(10):
            print('nop')
    try:
        expected = expected_fn()
    except Exception as e:
        with pytest.raises(e.__class__):
            actual_fn()
    else:
        assert expected == actual_fn()

def test_concatenated_sequence():
    if False:
        for i in range(10):
            print('nop')
    l1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    l2 = ConcatenatedSequence([0, 1], [], [2, 3, 4], [5, 6, 7, 8, 9], [])
    assert len(l1) == len(l2)
    for item in l1 + [999]:
        assert_equal_including_exceptions(lambda : l1.index(item), lambda : l2.index(item))
        for index in range(-15, 15):
            assert_equal_including_exceptions(lambda : l1.index(item, index), lambda : l2.index(item, index))
        for start_index in range(-15, 15):
            for end_index in range(-15, 15):
                assert_equal_including_exceptions(lambda : l1.index(item, start_index, end_index), lambda : l2.index(item, start_index, end_index))
    for index in range(-15, 15):
        assert_equal_including_exceptions(lambda : l1[index], lambda : l2[index])
    for start_index in range(-15, 15):
        for end_index in range(-15, 15):
            assert_equal_including_exceptions(lambda : l1[start_index:end_index], lambda : list(l2[start_index:end_index]))
    for item in l1 + [999]:
        assert_equal_including_exceptions(lambda : l1.count(item), lambda : l2.count(item))
    for item in l1 + [999]:
        assert_equal_including_exceptions(lambda : item in l1, lambda : item in l2)
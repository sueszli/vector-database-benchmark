from collections import OrderedDict
import pytest
from dvc.utils.humanize import get_summary, truncate_text

def test_get_summary():
    if False:
        return 10
    stats = OrderedDict([('fetched', 3), ('added', ['file1', 'file2', 'file3']), ('deleted', ['file4', 'file5']), ('modified', ['file6', 'file7'])])
    assert get_summary(stats.items()) == '3 files fetched, 3 files added, 2 files deleted and 2 files modified'
    del stats['fetched']
    del stats['deleted'][1]
    assert get_summary(stats.items()) == '3 files added, 1 file deleted and 2 files modified'
    del stats['deleted'][0]
    assert get_summary(stats.items()) == '3 files added and 2 files modified'
    del stats['modified']
    assert get_summary(stats.items()) == '3 files added'
    assert not get_summary([])
    assert not get_summary([('x', 0), ('y', [])])
    assert get_summary([('x', 1), ('y', [])]) == '1 file x'

def test_truncate_text():
    if False:
        return 10
    text = 'lorem ipsum'
    length = 5
    truncated = truncate_text(text, length)
    assert len(truncated) == length
    assert truncated[:-1] == text[:length - 1]
    assert truncated[-1] == '…'
    truncated = truncate_text(text, length, with_ellipsis=False)
    assert len(truncated) == length
    assert truncated == text[:length]

@pytest.mark.parametrize('with_ellipsis', [True, False])
def test_truncate_text_smaller_than_max_length(with_ellipsis):
    if False:
        i = 10
        return i + 15
    text = 'lorem ipsum'
    truncated = truncate_text(text, len(text), with_ellipsis=with_ellipsis)
    assert len(truncated) == len(text)
    assert truncated == text
    truncated = truncate_text(text, len(text) + 1, with_ellipsis=with_ellipsis)
    assert len(truncated) == len(text)
    assert truncated == text
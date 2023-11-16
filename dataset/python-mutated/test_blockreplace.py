import os
import textwrap
import pytest
import salt.utils.platform
import salt.utils.stringutils
pytestmark = [pytest.mark.windows_whitelisted]

class BlockreplaceParts:
    marker_start = '# start'
    marker_end = '# end'
    content = textwrap.dedent('        Line 1 of block\n        Line 2 of block\n        ')
    without_block = textwrap.dedent('        Hello world!\n\n        # comment here\n        ')
    with_non_matching_block = textwrap.dedent('        Hello world!\n\n        # start\n        No match here\n        # end\n        # comment here\n        ')
    with_non_matching_block_and_marker_end_not_after_newline = textwrap.dedent('        Hello world!\n\n        # start\n        No match here# end\n        # comment here\n        ')
    with_matching_block = textwrap.dedent('        Hello world!\n\n        # start\n        Line 1 of block\n        Line 2 of block\n        # end\n        # comment here\n        ')
    with_matching_block_and_extra_newline = textwrap.dedent('        Hello world!\n\n        # start\n        Line 1 of block\n        Line 2 of block\n\n        # end\n        # comment here\n        ')
    with_matching_block_and_marker_end_not_after_newline = textwrap.dedent('        Hello world!\n\n        # start\n        Line 1 of block\n        Line 2 of block# end\n        # comment here\n        ')
    content_explicit_posix_newlines = 'Line 1 of block\nLine 2 of block\n'
    content_explicit_windows_newlines = 'Line 1 of block\r\nLine 2 of block\r\n'
    without_block_explicit_posix_newlines = 'Hello world!\n\n# comment here\n'
    without_block_explicit_windows_newlines = 'Hello world!\r\n\r\n# comment here\r\n'
    with_block_prepended_explicit_posix_newlines = '# start\nLine 1 of block\nLine 2 of block\n# end\nHello world!\n\n# comment here\n'
    with_block_prepended_explicit_windows_newlines = '# start\r\nLine 1 of block\r\nLine 2 of block\r\n# end\r\nHello world!\r\n\r\n# comment here\r\n'
    with_block_appended_explicit_posix_newlines = 'Hello world!\n\n# comment here\n# start\nLine 1 of block\nLine 2 of block\n# end\n'
    with_block_appended_explicit_windows_newlines = 'Hello world!\r\n\r\n# comment here\r\n# start\r\nLine 1 of block\r\nLine 2 of block\r\n# end\r\n'

def strip_ending_linebreak_ids(value):
    if False:
        while True:
            i = 10
    return 'strip_ending_linebreak={}'.format(value)

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_prepend(file, tmp_path, strip_ending_linebreak):
    if False:
        return 10
    "\n    Test blockreplace when prepend_if_not_found=True and block doesn't\n    exist in file.\n    "
    name = tmp_path / 'testfile'
    expected = BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n' + BlockreplaceParts.without_block
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_prepend_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        i = 10
        return i + 15
    "\n    Test blockreplace when prepend_if_not_found=True and block doesn't\n    exist in file. Test with append_newline explicitly set to True.\n    "
    name = tmp_path / 'testfile'
    if strip_ending_linebreak:
        expected = BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n' + BlockreplaceParts.without_block
    else:
        expected = BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + '\n' + BlockreplaceParts.marker_end + '\n' + BlockreplaceParts.without_block
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True, append_newline=True)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_prepend_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    "\n    Test blockreplace when prepend_if_not_found=True and block doesn't\n    exist in file. Test with append_newline explicitly set to False.\n    "
    name = tmp_path / 'testfile'
    if strip_ending_linebreak:
        expected = BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content.rstrip('\r\n') + BlockreplaceParts.marker_end + '\n' + BlockreplaceParts.without_block
    else:
        expected = BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n' + BlockreplaceParts.without_block
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True, append_newline=False)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_append(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    "\n    Test blockreplace when append_if_not_found=True and block doesn't\n    exist in file.\n    "
    name = tmp_path / 'testfile'
    expected = BlockreplaceParts.without_block + BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n'
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_append_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        for i in range(10):
            print('nop')
    "\n    Test blockreplace when append_if_not_found=True and block doesn't\n    exist in file. Test with append_newline explicitly set to True.\n    "
    name = tmp_path / 'testfile'
    if strip_ending_linebreak:
        expected = BlockreplaceParts.without_block + BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n'
    else:
        expected = BlockreplaceParts.without_block + BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + '\n' + BlockreplaceParts.marker_end + '\n'
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True, append_newline=True)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_append_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    "\n    Test blockreplace when append_if_not_found=True and block doesn't\n    exist in file. Test with append_newline explicitly set to False.\n    "
    name = tmp_path / 'testfile'
    if strip_ending_linebreak:
        expected = BlockreplaceParts.without_block + BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content.rstrip('\r\n') + BlockreplaceParts.marker_end + '\n'
    else:
        expected = BlockreplaceParts.without_block + BlockreplaceParts.marker_start + '\n' + BlockreplaceParts.content + BlockreplaceParts.marker_end + '\n'
    name.write_text(BlockreplaceParts.without_block)
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True, append_newline=False)
    assert ret.result is True
    assert ret.changes
    contents = name.read_text()
    assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    contents = name.read_text()
    assert contents == expected

def line_breaks_ids(value):
    if False:
        print('Hello World!')
    return 'line_breaks={}'.format(value)

@pytest.mark.parametrize('line_breaks', ('windows', 'posix'), ids=line_breaks_ids)
def test_prepend_auto_line_separator(file, tmp_path, line_breaks):
    if False:
        i = 10
        return i + 15
    '\n    This tests the line separator auto-detection when prepending the block\n    '
    name = tmp_path / 'testfile'
    if line_breaks == 'posix':
        name.write_text(BlockreplaceParts.without_block_explicit_windows_newlines)
        content = BlockreplaceParts.content_explicit_posix_newlines
        expected = BlockreplaceParts.with_block_prepended_explicit_windows_newlines
    else:
        name.write_text(BlockreplaceParts.without_block_explicit_posix_newlines)
        content = BlockreplaceParts.content_explicit_windows_newlines
        expected = BlockreplaceParts.with_block_prepended_explicit_posix_newlines
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, prepend_if_not_found=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('line_breaks', ('windows', 'posix'), ids=line_breaks_ids)
def test_append_auto_line_separator(file, tmp_path, line_breaks):
    if False:
        for i in range(10):
            print('nop')
    '\n    This tests the line separator auto-detection when appending the block\n    '
    name = tmp_path / 'testfile'
    if line_breaks == 'posix':
        name.write_text(BlockreplaceParts.without_block_explicit_windows_newlines)
        content = BlockreplaceParts.content_explicit_posix_newlines
        expected = BlockreplaceParts.with_block_appended_explicit_windows_newlines
    else:
        name.write_text(BlockreplaceParts.without_block_explicit_posix_newlines)
        content = BlockreplaceParts.content_explicit_windows_newlines
        expected = BlockreplaceParts.with_block_appended_explicit_posix_newlines
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_if_not_found=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block(file, tmp_path, strip_ending_linebreak):
    if False:
        print('Hello World!')
    '\n    Test blockreplace when block exists but its contents are not a match.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block)
    expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        return 10
    '\n    Test blockreplace when block exists but its contents are not a\n    match. Test with append_newline explicitly set to True.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block
    else:
        expected = BlockreplaceParts.with_matching_block_and_extra_newline
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        return 10
    '\n    Test blockreplace when block exists but its contents are not a\n    match. Test with append_newline explicitly set to False.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline
    else:
        expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block_and_marker_not_after_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    '\n    Test blockreplace when block exists but its contents are not a\n    match, and the marker_end is not directly preceded by a newline.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block_and_marker_end_not_after_newline)
    expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block_and_marker_not_after_newline_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    '\n    Test blockreplace when block exists but its contents are not a match,\n    and the marker_end is not directly preceded by a newline. Test with\n    append_newline explicitly set to True.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block_and_marker_end_not_after_newline)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block
    else:
        expected = BlockreplaceParts.with_matching_block_and_extra_newline
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_non_matching_block_and_marker_not_after_newline_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    '\n    Test blockreplace when block exists but its contents are not a match,\n    and the marker_end is not directly preceded by a newline. Test with\n    append_newline explicitly set to False.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_non_matching_block_and_marker_end_not_after_newline)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline
    else:
        expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block(file, tmp_path, strip_ending_linebreak):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test blockreplace when block exists and its contents are a match. No\n    changes should be made.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block)
    expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        i = 10
        return i + 15
    '\n    Test blockreplace when block exists and its contents are a match. Test\n    with append_newline explicitly set to True. This will result in an\n    extra newline when the content ends in a newline, and will not when the\n    content does not end in a newline.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block
    else:
        expected = BlockreplaceParts.with_matching_block_and_extra_newline
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    if strip_ending_linebreak:
        assert not ret.changes
    else:
        assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        i = 10
        return i + 15
    '\n    Test blockreplace when block exists and its contents are a match. Test\n    with append_newline explicitly set to False. This will result in the\n    marker_end not being directly preceded by a newline when the content\n    does not end in a newline.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline
    else:
        expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    if strip_ending_linebreak:
        assert ret.changes
    else:
        assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block_and_marker_not_after_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        print('Hello World!')
    '\n    Test blockreplace when block exists and its contents are a match, but\n    the marker_end is not directly preceded by a newline.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline)
    expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block_and_marker_not_after_newline_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        i = 10
        return i + 15
    '\n    Test blockreplace when block exists and its contents are a match, but\n    the marker_end is not directly preceded by a newline. Test with\n    append_newline explicitly set to True. This will result in an extra\n    newline when the content ends in a newline, and will not when the\n    content does not end in a newline.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block
    else:
        expected = BlockreplaceParts.with_matching_block_and_extra_newline
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=True)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

@pytest.mark.parametrize('strip_ending_linebreak', (False, True), ids=strip_ending_linebreak_ids)
def test_matching_block_and_marker_not_after_newline_no_append_newline(file, tmp_path, strip_ending_linebreak):
    if False:
        while True:
            i = 10
    '\n    Test blockreplace when block exists and its contents are a match, but\n    the marker_end is not directly preceded by a newline. Test with\n    append_newline explicitly set to False.\n    '
    name = tmp_path / 'testfile'
    name.write_text(BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline)
    if strip_ending_linebreak:
        expected = BlockreplaceParts.with_matching_block_and_marker_end_not_after_newline
    else:
        expected = BlockreplaceParts.with_matching_block
    content = BlockreplaceParts.content
    if strip_ending_linebreak:
        content = content.rstrip('\r\n')
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    if strip_ending_linebreak:
        assert not ret.changes
    else:
        assert ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected
    ret = file.blockreplace(name=str(name), content=content, marker_start=BlockreplaceParts.marker_start, marker_end=BlockreplaceParts.marker_end, append_newline=False)
    assert ret.result is True
    assert not ret.changes
    if not salt.utils.platform.is_windows():
        contents = salt.utils.stringutils.to_unicode(name.read_bytes())
        assert contents == expected

def test_issue_49043(file, tmp_path, state_tree):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test blockreplace with jinja template and unicode strings in the context\n    '
    if salt.utils.platform.is_windows() and os.environ.get('PYTHONUTF8', '0') == '0':
        pytest.skip('Test will fail if PYTHONUTF8=1 is not set on windows')
    name = tmp_path / 'testfile'
    name.touch()
    unicode_string = 'äöü'
    expected = textwrap.dedent('        #-- start managed zone --\n        {}\n        #-- end managed zone --\n        '.format(unicode_string))
    with pytest.helpers.temp_file('issue-49043', directory=state_tree, contents='{{ unicode_string }}'):
        ret = file.blockreplace(name=str(name), source='salt://issue-49043', append_if_not_found=True, template='jinja', context={'unicode_string': unicode_string})
        assert ret.result is True
        assert name.read_text() == expected
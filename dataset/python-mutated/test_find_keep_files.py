import logging
import pytest
import salt.states.file as filestate
log = logging.getLogger(__name__)

@pytest.mark.skip_on_windows(reason='Do not run on Windows')
def test__find_keep_files_unix():
    if False:
        print('Hello World!')
    keep = filestate._find_keep_files('/test/parent_folder', ['/test/parent_folder/meh.txt'])
    expected = ['/', '/test', '/test/parent_folder', '/test/parent_folder/meh.txt']
    actual = sorted(list(keep))
    assert actual == expected, actual

@pytest.mark.skip_unless_on_windows(reason='Do not run except on Windows')
def test__find_keep_files_win32():
    if False:
        while True:
            i = 10
    '\n    Test _find_keep_files. The `_find_keep_files` function is only called by\n    _clean_dir.\n    '
    keep = filestate._find_keep_files('c:\\test\\parent_folder', ['C:\\test\\parent_folder\\meh-1.txt', 'C:\\Test\\Parent_folder\\Meh-2.txt'])
    expected = ['c:\\', 'c:\\test', 'c:\\test\\parent_folder', 'c:\\test\\parent_folder\\meh-1.txt', 'c:\\test\\parent_folder\\meh-2.txt']
    actual = sorted(list(keep))
    assert actual == expected

@pytest.mark.skip_unless_on_windows(reason='Do not run except on Windows')
def test__clean_dir_win32():
    if False:
        while True:
            i = 10
    '\n    Test _clean_dir to ensure that regardless of case, we keep all files\n    requested and do not delete any. Therefore, the expected list should\n    be empty for this test.\n    '
    keep = filestate._clean_dir('c:\\test\\parent_folder', ['C:\\test\\parent_folder\\meh-1.txt', 'C:\\Test\\Parent_folder\\Meh-2.txt'], exclude_pat=None)
    actual = sorted(list(keep))
    expected = []
    assert actual == expected

@pytest.mark.skip_unless_on_darwin(reason='Do not run except on OS X')
def test__find_keep_files_darwin():
    if False:
        print('Hello World!')
    '\n    Test _clean_dir to ensure that regardless of case, we keep all files\n    requested and do not delete any. Therefore, the expected list should\n    be empty for this test.\n    '
    keep = filestate._clean_dir('/test/parent_folder', ['/test/folder/parent_folder/meh-1.txt', '/Test/folder/Parent_Folder/Meh-2.txt'], exclude_pat=None)
    actual = sorted(list(keep))
    expected = []
    assert actual == expected
from __future__ import annotations
import stat
from airflow_breeze.utils.run_utils import change_directory_permission, change_file_permission, filter_out_none

def test_change_file_permission(tmp_path):
    if False:
        i = 10
        return i + 15
    tmpfile = tmp_path / 'test.config'
    tmpfile.write_text('content')
    change_file_permission(tmpfile)
    mode = tmpfile.stat().st_mode
    assert not mode & stat.S_IWGRP and (not mode & stat.S_IWOTH)

def test_change_directory_permission(tmp_path):
    if False:
        while True:
            i = 10
    subdir = tmp_path / 'testdir'
    subdir.mkdir()
    change_directory_permission(subdir)
    mode = subdir.stat().st_mode
    assert not mode & stat.S_IWGRP and (not mode & stat.S_IWOTH) and mode & stat.S_IXGRP and mode & stat.S_IXOTH

def test_filter_out_none():
    if False:
        for i in range(10):
            print('nop')
    dict_input_with_none = {'sample': None, 'sample1': 'One', 'sample2': 'Two', 'samplen': None}
    expected_dict_output = {'sample1': 'One', 'sample2': 'Two'}
    output_dict = filter_out_none(**dict_input_with_none)
    assert output_dict == expected_dict_output
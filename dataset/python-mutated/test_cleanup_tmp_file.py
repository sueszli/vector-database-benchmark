from __future__ import annotations
import os
import tempfile
from ansible.utils.path import cleanup_tmp_file

def test_cleanup_tmp_file_file():
    if False:
        return 10
    (tmp_fd, tmp) = tempfile.mkstemp()
    cleanup_tmp_file(tmp)
    assert not os.path.exists(tmp)

def test_cleanup_tmp_file_dir():
    if False:
        return 10
    tmp = tempfile.mkdtemp()
    cleanup_tmp_file(tmp)
    assert not os.path.exists(tmp)

def test_cleanup_tmp_file_nonexistant():
    if False:
        return 10
    assert None is cleanup_tmp_file('nope')

def test_cleanup_tmp_file_failure(mocker, capsys):
    if False:
        return 10
    tmp = tempfile.mkdtemp()
    rmtree = mocker.patch('shutil.rmtree', side_effect=OSError('test induced failure'))
    cleanup_tmp_file(tmp)
    (out, err) = capsys.readouterr()
    assert out == ''
    assert err == ''
    rmtree.assert_called_once()

def test_cleanup_tmp_file_failure_warning(mocker, capsys):
    if False:
        for i in range(10):
            print('nop')
    tmp = tempfile.mkdtemp()
    rmtree = mocker.patch('shutil.rmtree', side_effect=OSError('test induced failure'))
    cleanup_tmp_file(tmp, warn=True)
    (out, err) = capsys.readouterr()
    assert out == 'Unable to remove temporary file test induced failure\n'
    assert err == ''
    rmtree.assert_called_once()
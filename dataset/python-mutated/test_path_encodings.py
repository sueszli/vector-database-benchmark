import os
import subprocess
import pytest
from PyInstaller.utils.tests import importorskip

def test_ascii_path(pyi_builder):
    if False:
        i = 10
        return i + 15
    distdir = pyi_builder._distdir
    dd_ascii = distdir.encode('ascii', 'replace').decode('ascii')
    if distdir != dd_ascii:
        pytest.skip(reason='Default build path not ASCII, skipping...')
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.linux
def test_linux_non_unicode_path(pyi_builder, monkeypatch):
    if False:
        while True:
            i = 10
    distdir = pyi_builder._distdir
    unicode_filename = 'ěščřžýáíé日本語'
    pyi_builder._distdir = os.path.join(distdir, unicode_filename)
    os.makedirs(pyi_builder._distdir)
    tmpdir = os.path.join(str(pyi_builder._tmpdir), unicode_filename + '_TMP')
    monkeypatch.setenv('LC_ALL', 'C')
    monkeypatch.setenv('TMPDIR', tmpdir)
    monkeypatch.setenv('TMP', tmpdir)
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.darwin
@pytest.mark.linux
def test_osx_linux_unicode_path(pyi_builder, monkeypatch):
    if False:
        print('Hello World!')
    distdir = pyi_builder._distdir
    unicode_filename = 'ěščřžýáíé日本語'
    pyi_builder._distdir = os.path.join(distdir, unicode_filename)
    os.makedirs(pyi_builder._distdir)
    tmpdir = os.path.join(str(pyi_builder._tmpdir), unicode_filename + '_TMP')
    monkeypatch.setenv('TMPDIR', tmpdir)
    monkeypatch.setenv('TMP', tmpdir)
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.win32
def test_win_codepage_path(pyi_builder, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    distdir = pyi_builder._distdir
    cp_filename = bytes(bytearray(range(128, 134))).decode('mbcs')
    pyi_builder._distdir = os.path.join(distdir, cp_filename)
    os.makedirs(pyi_builder._distdir)
    tmpdir = os.path.join(str(pyi_builder._tmpdir), cp_filename + '_TMP')
    monkeypatch.setenv('TMPDIR', tmpdir)
    monkeypatch.setenv('TMP', tmpdir)
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.win32
def test_win_codepage_path_disabled_shortfilename(pyi_builder, monkeypatch):
    if False:
        i = 10
        return i + 15
    distdir = pyi_builder._distdir
    cp_filename = bytes(bytearray(range(128, 134))).decode('mbcs')
    distdir = os.path.join(distdir, cp_filename)
    os.makedirs(distdir)
    fsutil_distdir = pyi_builder._distdir
    if subprocess.call(['fsutil', '8dot3name', 'strip', fsutil_distdir]):
        pytest.xfail('Administrator privileges required to strip ShortFileName.')
    tmpdir = os.path.join(str(pyi_builder._tmpdir), cp_filename + '_TMP')
    monkeypatch.setenv('TMPDIR', tmpdir)
    monkeypatch.setenv('TMP', tmpdir)
    pyi_builder._distdir = distdir
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.win32
def test_win_non_codepage_path(pyi_builder, monkeypatch):
    if False:
        return 10
    distdir = pyi_builder._distdir
    non_cp_filename = 'ěščřžýáíé日本語'
    pyi_builder._distdir = os.path.join(distdir, non_cp_filename)
    os.makedirs(pyi_builder._distdir)
    tmpdir = os.path.join(str(pyi_builder._tmpdir), non_cp_filename + '_TMP')
    monkeypatch.setenv('TMPDIR', tmpdir)
    monkeypatch.setenv('TMP', tmpdir)
    pyi_builder.test_script('pyi_path_encoding.py')

@pytest.mark.win32
@importorskip('win32api')
def test_win_py3_no_shortpathname(pyi_builder):
    if False:
        print('Hello World!')
    pyi_builder.test_script('pyi_win_py3_no_shortpathname.py')

@pytest.mark.win32
@importorskip('win32api')
def test_win_TEMP_has_shortpathname(pyi_builder, monkeypatch, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test if script if pass if $TMP holds a short path name.\n    '
    tmp = tmp_path / 'longlongfilename' / 'xxx'
    tmp.mkdir(parents=True, exist_ok=True)
    import win32api
    tmp = win32api.GetShortPathName(str(tmp))
    monkeypatch.setenv('TMP', tmp)
    monkeypatch.setenv('TEMP', tmp)
    pyi_builder.test_script('pyi_win_py3_no_shortpathname.py')
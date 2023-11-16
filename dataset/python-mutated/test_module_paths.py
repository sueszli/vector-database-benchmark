"""Tests for IPython.utils.module_paths.py"""
import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
TEST_FILE_PATH = Path(__file__).resolve().parent
TMP_TEST_DIR = Path(tempfile.mkdtemp(suffix='with.dot'))
old_syspath = sys.path

def make_empty_file(fname):
    if False:
        return 10
    open(fname, 'w', encoding='utf-8').close()

def setup_module():
    if False:
        i = 10
        return i + 15
    'Setup testenvironment for the module:\n\n    '
    Path(TMP_TEST_DIR / 'xmod').mkdir(parents=True)
    Path(TMP_TEST_DIR / 'nomod').mkdir(parents=True)
    make_empty_file(TMP_TEST_DIR / 'xmod/__init__.py')
    make_empty_file(TMP_TEST_DIR / 'xmod/sub.py')
    make_empty_file(TMP_TEST_DIR / 'pack.py')
    make_empty_file(TMP_TEST_DIR / 'packpyc.pyc')
    sys.path = [str(TMP_TEST_DIR)]

def teardown_module():
    if False:
        while True:
            i = 10
    'Teardown testenvironment for the module:\n\n    - Remove tempdir\n    - restore sys.path\n    '
    shutil.rmtree(TMP_TEST_DIR)
    sys.path = old_syspath

def test_tempdir():
    if False:
        i = 10
        return i + 15
    '\n    Ensure the test are done with a temporary file that have a dot somewhere.\n    '
    assert '.' in str(TMP_TEST_DIR)

def test_find_mod_1():
    if False:
        return 10
    "\n    Search for a directory's file path.\n    Expected output: a path to that directory's __init__.py file.\n    "
    modpath = TMP_TEST_DIR / 'xmod' / '__init__.py'
    assert Path(mp.find_mod('xmod')) == modpath

def test_find_mod_2():
    if False:
        i = 10
        return i + 15
    "\n    Search for a directory's file path.\n    Expected output: a path to that directory's __init__.py file.\n    TODO: Confirm why this is a duplicate test.\n    "
    modpath = TMP_TEST_DIR / 'xmod' / '__init__.py'
    assert Path(mp.find_mod('xmod')) == modpath

def test_find_mod_3():
    if False:
        for i in range(10):
            print('nop')
    '\n    Search for a directory + a filename without its .py extension\n    Expected output: full path with .py extension.\n    '
    modpath = TMP_TEST_DIR / 'xmod' / 'sub.py'
    assert Path(mp.find_mod('xmod.sub')) == modpath

def test_find_mod_4():
    if False:
        print('Hello World!')
    '\n    Search for a filename without its .py extension\n    Expected output: full path with .py extension\n    '
    modpath = TMP_TEST_DIR / 'pack.py'
    assert Path(mp.find_mod('pack')) == modpath

def test_find_mod_5():
    if False:
        for i in range(10):
            print('nop')
    '\n    Search for a filename with a .pyc extension\n    Expected output: TODO: do we exclude or include .pyc files?\n    '
    assert mp.find_mod('packpyc') == None
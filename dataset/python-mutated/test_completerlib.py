"""Tests for completerlib.

"""
import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths

class MockEvent(object):

    def __init__(self, line):
        if False:
            while True:
                i = 10
        self.line = line

class Test_magic_run_completer(unittest.TestCase):
    files = [u'aao.py', u'a.py', u'b.py', u'aao.txt']
    dirs = [u'adir/', 'bdir/']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in self.files:
            with open(join(self.BASETESTDIR, fil), 'w', encoding='utf-8') as sfile:
                sfile.write('pass\n')
        for d in self.dirs:
            os.mkdir(join(self.BASETESTDIR, d))
        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        if False:
            while True:
                i = 10
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    def test_1(self):
        if False:
            print('Hello World!')
        'Test magic_run_completer, should match two alternatives\n        '
        event = MockEvent(u'%run a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aao.py', u'adir/'})

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test magic_run_completer, should match one alternative\n        '
        event = MockEvent(u'%run aa')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'aao.py'})

    def test_3(self):
        if False:
            print('Hello World!')
        'Test magic_run_completer with unterminated " '
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aao.py', u'adir/'})

    def test_completion_more_args(self):
        if False:
            return 10
        event = MockEvent(u'%run a.py ')
        match = set(magic_run_completer(None, event))
        self.assertEqual(match, set(self.files + self.dirs))

    def test_completion_in_dir(self):
        if False:
            while True:
                i = 10
        event = MockEvent(u'%run a.py {}'.format(join(self.BASETESTDIR, 'a')))
        print(repr(event.line))
        match = set(magic_run_completer(None, event))
        self.assertEqual(match, {join(self.BASETESTDIR, f).replace('\\', '/') for f in (u'a.py', u'aao.py', u'aao.txt', u'adir/')})

class Test_magic_run_completer_nonascii(unittest.TestCase):

    @onlyif_unicode_paths
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in [u'aaø.py', u'a.py', u'b.py']:
            with open(join(self.BASETESTDIR, fil), 'w', encoding='utf-8') as sfile:
                sfile.write('pass\n')
        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        if False:
            print('Hello World!')
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        if False:
            print('Hello World!')
        'Test magic_run_completer, should match two alternatives\n        '
        event = MockEvent(u'%run a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aaø.py'})

    @onlyif_unicode_paths
    def test_2(self):
        if False:
            return 10
        'Test magic_run_completer, should match one alternative\n        '
        event = MockEvent(u'%run aa')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'aaø.py'})

    @onlyif_unicode_paths
    def test_3(self):
        if False:
            while True:
                i = 10
        'Test magic_run_completer with unterminated " '
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aaø.py'})

def test_import_invalid_module():
    if False:
        print('Hello World!')
    'Testing of issue https://github.com/ipython/ipython/issues/1107'
    invalid_module_names = {'foo-bar', 'foo:bar', '10foo'}
    valid_module_names = {'foobar'}
    with TemporaryDirectory() as tmpdir:
        sys.path.insert(0, tmpdir)
        for name in invalid_module_names | valid_module_names:
            filename = os.path.join(tmpdir, name + '.py')
            open(filename, 'w', encoding='utf-8').close()
        s = set(module_completion('import foo'))
        intersection = s.intersection(invalid_module_names)
        assert intersection == set()
        assert valid_module_names.issubset(s), valid_module_names.intersection(s)

def test_bad_module_all():
    if False:
        for i in range(10):
            print('nop')
    'Test module with invalid __all__\n\n    https://github.com/ipython/ipython/issues/9678\n    '
    testsdir = os.path.dirname(__file__)
    sys.path.insert(0, testsdir)
    try:
        results = module_completion('from bad_all import ')
        assert 'puppies' in results
        for r in results:
            assert isinstance(r, str)
        results = module_completion('import bad_all.')
        assert results == []
    finally:
        sys.path.remove(testsdir)

def test_module_without_init():
    if False:
        while True:
            i = 10
    '\n    Test module without __init__.py.\n    \n    https://github.com/ipython/ipython/issues/11226\n    '
    fake_module_name = 'foo'
    with TemporaryDirectory() as tmpdir:
        sys.path.insert(0, tmpdir)
        try:
            os.makedirs(os.path.join(tmpdir, fake_module_name))
            s = try_import(mod=fake_module_name)
            assert s == [], f'for module {fake_module_name}'
        finally:
            sys.path.remove(tmpdir)

def test_valid_exported_submodules():
    if False:
        return 10
    '\n    Test checking exported (__all__) objects are submodules\n    '
    results = module_completion('import os.pa')
    assert 'os.path' in results
    assert 'os.pathconf' not in results
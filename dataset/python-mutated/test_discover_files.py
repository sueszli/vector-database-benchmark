from __future__ import annotations
import os.path
import pytest
from flake8 import utils
from flake8.discover_files import _filenames_from
from flake8.discover_files import expand_paths

@pytest.fixture
def files_dir(tmpdir):
    if False:
        i = 10
        return i + 15
    'Create test dir for testing filenames_from.'
    with tmpdir.as_cwd():
        tmpdir.join('a/b/c.py').ensure()
        tmpdir.join('a/b/d.py').ensure()
        tmpdir.join('a/b/e/f.py').ensure()
        yield tmpdir

def _noop(path):
    if False:
        return 10
    return False

def _normpath(s):
    if False:
        return 10
    return s.replace('/', os.sep)

def _normpaths(pths):
    if False:
        while True:
            i = 10
    return {_normpath(pth) for pth in pths}

@pytest.mark.usefixtures('files_dir')
def test_filenames_from_a_directory():
    if False:
        i = 10
        return i + 15
    'Verify that filenames_from walks a directory.'
    filenames = set(_filenames_from(_normpath('a/b/'), predicate=_noop))
    expected = _normpaths(('a/b/c.py', 'a/b/d.py', 'a/b/e/f.py'))
    assert filenames == expected

@pytest.mark.usefixtures('files_dir')
def test_filenames_from_a_directory_with_a_predicate():
    if False:
        print('Hello World!')
    'Verify that predicates filter filenames_from.'
    filenames = set(_filenames_from(arg=_normpath('a/b/'), predicate=lambda path: path.endswith(_normpath('b/c.py'))))
    expected = _normpaths(('a/b/d.py', 'a/b/e/f.py'))
    assert filenames == expected

@pytest.mark.usefixtures('files_dir')
def test_filenames_from_a_directory_with_a_predicate_from_the_current_dir():
    if False:
        print('Hello World!')
    'Verify that predicates filter filenames_from.'
    filenames = set(_filenames_from(arg=_normpath('./a/b'), predicate=lambda path: path == 'c.py'))
    expected = _normpaths(('./a/b/c.py', './a/b/d.py', './a/b/e/f.py'))
    assert filenames == expected

@pytest.mark.usefixtures('files_dir')
def test_filenames_from_a_single_file():
    if False:
        for i in range(10):
            print('nop')
    'Verify that we simply yield that filename.'
    filenames = set(_filenames_from(_normpath('a/b/c.py'), predicate=_noop))
    assert filenames == {_normpath('a/b/c.py')}

def test_filenames_from_a_single_file_does_not_exist():
    if False:
        i = 10
        return i + 15
    'Verify that a passed filename which does not exist is returned back.'
    filenames = set(_filenames_from(_normpath('d/n/e.py'), predicate=_noop))
    assert filenames == {_normpath('d/n/e.py')}

def test_filenames_from_exclude_doesnt_exclude_directory_names(tmpdir):
    if False:
        return 10
    "Verify that we don't greedily exclude subdirs."
    tmpdir.join('1/dont_return_me.py').ensure()
    tmpdir.join('2/1/return_me.py').ensure()
    exclude = [tmpdir.join('1').strpath]

    def predicate(pth):
        if False:
            while True:
                i = 10
        return utils.fnmatch(os.path.abspath(pth), exclude)
    with tmpdir.as_cwd():
        filenames = list(_filenames_from('.', predicate=predicate))
    assert filenames == [os.path.join('.', '2', '1', 'return_me.py')]

def test_filenames_from_predicate_applies_to_initial_arg(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that the predicate is also applied to the passed argument.'
    fname = str(tmp_path.joinpath('f.py'))
    ret = tuple(_filenames_from(fname, predicate=lambda _: True))
    assert ret == ()

def test_filenames_from_predicate_applies_to_dirname(tmp_path):
    if False:
        while True:
            i = 10
    'Test that the predicate can filter whole directories.'
    a_dir = tmp_path.joinpath('a')
    a_dir.mkdir()
    a_dir.joinpath('b.py').touch()
    b_py = tmp_path.joinpath('b.py')
    b_py.touch()

    def predicate(p):
        if False:
            while True:
                i = 10
        return p.endswith('a')
    ret = tuple(_filenames_from(str(tmp_path), predicate=predicate))
    assert ret == (str(b_py),)

def _expand_paths(*, paths=('.',), stdin_display_name='stdin', filename_patterns=('*.py',), exclude=()):
    if False:
        print('Hello World!')
    return set(expand_paths(paths=paths, stdin_display_name=stdin_display_name, filename_patterns=filename_patterns, exclude=exclude))

@pytest.mark.usefixtures('files_dir')
def test_expand_paths_honors_exclude():
    if False:
        return 10
    expected = _normpaths(('./a/b/c.py', './a/b/e/f.py'))
    assert _expand_paths(exclude=['d.py']) == expected

@pytest.mark.usefixtures('files_dir')
def test_expand_paths_defaults_to_dot():
    if False:
        print('Hello World!')
    expected = _normpaths(('./a/b/c.py', './a/b/d.py', './a/b/e/f.py'))
    assert _expand_paths(paths=()) == expected

def test_default_stdin_name_is_not_filtered():
    if False:
        i = 10
        return i + 15
    assert _expand_paths(paths=('-',)) == {'-'}

def test_alternate_stdin_name_is_filtered():
    if False:
        print('Hello World!')
    ret = _expand_paths(paths=('-',), stdin_display_name='wat', exclude=('wat',))
    assert ret == set()

def test_filename_included_even_if_not_matching_include(tmp_path):
    if False:
        return 10
    some_file = str(tmp_path.joinpath('some/file'))
    assert _expand_paths(paths=(some_file,)) == {some_file}
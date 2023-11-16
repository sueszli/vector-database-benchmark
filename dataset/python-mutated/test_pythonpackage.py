"""
THESE TESTS DON'T RUN IN GITHUB-ACTIONS (takes too long!!)
ONLY THE BASIC ONES IN test_pythonpackage_basic.py DO.

(This file basically covers all tests for any of the
functions that aren't already part of the basic
test set)
"""
import os
import shutil
import tempfile
from pythonforandroid.pythonpackage import _extract_info_from_package, extract_metainfo_files_from_package, get_package_as_folder, get_package_dependencies

def local_repo_folder():
    if False:
        while True:
            i = 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def test_get_package_dependencies():
    if False:
        i = 10
        return i + 15
    deps_nonrecursive = get_package_dependencies(local_repo_folder(), recursive=False)
    deps_recursive = get_package_dependencies(local_repo_folder(), recursive=True)
    assert len([dep for dep in deps_nonrecursive if 'jinja2' in dep]) > 0
    assert [dep for dep in deps_recursive if 'MarkupSafe' in dep]
    assert 'setuptools' not in deps_nonrecursive
    assert 'setuptools' in get_package_dependencies(local_repo_folder(), recursive=False, include_build_requirements=True)
    assert len([dep for dep in get_package_dependencies('python-for-android') if 'jinja2' in dep]) > 0
    assert [dep for dep in get_package_dependencies('python-for-android', recursive=True) if 'MarkupSafe' in dep]

def test_extract_metainfo_files_from_package():
    if False:
        print('Hello World!')
    files_dir = tempfile.mkdtemp()
    try:
        extract_metainfo_files_from_package('python-for-android', files_dir, debug=True)
        assert os.path.exists(os.path.join(files_dir, 'METADATA'))
    finally:
        shutil.rmtree(files_dir)
    files_dir = tempfile.mkdtemp()
    try:
        extract_metainfo_files_from_package(local_repo_folder(), files_dir, debug=True)
        assert os.path.exists(os.path.join(files_dir, 'METADATA'))
    finally:
        shutil.rmtree(files_dir)

def test_get_package_as_folder():
    if False:
        return 10
    (obtained_type, obtained_path) = get_package_as_folder('python-for-android')
    try:
        assert obtained_type in {'source', 'wheel'}
        assert os.path.isdir(obtained_path)
    finally:
        shutil.rmtree(obtained_path)

def test__extract_info_from_package():
    if False:
        for i in range(10):
            print('nop')
    assert _extract_info_from_package(local_repo_folder(), extract_type='name') == 'python-for-android'
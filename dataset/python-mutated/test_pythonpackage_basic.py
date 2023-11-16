"""
ONLY BASIC TEST SET. The additional ones are in test_pythonpackage.py.

These are in a separate file because these were picked to run in github-actions,
while the other additional ones aren't (for build time reasons).
"""
import os
import shutil
import sys
import subprocess
import tempfile
import textwrap
from unittest import mock
from pythonforandroid.pythonpackage import _extract_info_from_package, get_dep_names_of_package, get_package_name, _get_system_python_executable, is_filesystem_path, parse_as_folder_reference, transform_dep_for_pip

def local_repo_folder():
    if False:
        while True:
            i = 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def fake_metadata_extract(dep_name, output_folder, debug=False):
    if False:
        return 10
    with open(os.path.join(output_folder, 'METADATA'), 'w') as f:
        f.write(textwrap.dedent('            Metadata-Version: 2.1\n            Name: testpackage\n            Version: 0.1\n            Requires-Dist: testpkg\n            Requires-Dist: testpkg2\n\n            Lorem Ipsum'))
    with open(os.path.join(output_folder, 'metadata_source'), 'w') as f:
        f.write(u'wheel')

def test__extract_info_from_package():
    if False:
        return 10
    import pythonforandroid.pythonpackage
    with mock.patch('pythonforandroid.pythonpackage.extract_metainfo_files_from_package', fake_metadata_extract):
        assert _extract_info_from_package('whatever', extract_type='name') == 'testpackage'
        assert set(_extract_info_from_package('whatever', extract_type='dependencies')) == {'testpkg', 'testpkg2'}

def test_get_package_name():
    if False:
        while True:
            i = 10
    with mock.patch('pythonforandroid.pythonpackage.extract_metainfo_files_from_package', fake_metadata_extract):
        assert get_package_name('TeStPackaGe') == 'testpackage'
    temp_d = tempfile.mkdtemp(prefix='p4a-pythonpackage-test-tmp-')
    try:
        with open(os.path.join(temp_d, 'setup.py'), 'w') as f:
            f.write(textwrap.dedent('                from setuptools import setup\n                setup(name="testpackage")\n                '))
        pkg_name = get_package_name(temp_d)
        assert pkg_name == 'testpackage'
    finally:
        shutil.rmtree(temp_d)

def test_get_dep_names_of_package():
    if False:
        print('Hello World!')
    dep_names = get_dep_names_of_package('python-for-android')
    assert 'colorama' in dep_names
    assert 'setuptools' not in dep_names
    try:
        dep_names = get_dep_names_of_package('python-for-android', include_build_requirements=True, verbose=True)
    except NotImplementedError as e:
        assert 'wheel' in str(e)
    else:
        assert 'setuptools' in dep_names
    assert 'colorama' in get_dep_names_of_package(local_repo_folder())
    test_fake_package = tempfile.mkdtemp()
    try:
        with open(os.path.join(test_fake_package, 'setup.py'), 'w') as f:
            f.write(textwrap.dedent("            from setuptools import setup\n\n            setup(name='fakeproject',\n                  description='fake for testing',\n                  install_requires=['buildozer==0.39',\n                                    'python-for-android>=0.5.1'],\n            )\n            "))
        assert set(get_dep_names_of_package(test_fake_package, recursive=False, keep_version_pins=True, verbose=True)) == {'buildozer==0.39', 'python-for-android'}
        assert set(get_dep_names_of_package(test_fake_package, recursive=False, keep_version_pins=False, verbose=True)) == {'buildozer', 'python-for-android'}
        dep_names = get_dep_names_of_package(test_fake_package, recursive=False, keep_version_pins=False, verbose=True, include_build_requirements=True)
        assert len({'buildozer', 'python-for-android', 'setuptools'}.intersection(dep_names)) == 3
    finally:
        shutil.rmtree(test_fake_package)

def test_transform_dep_for_pip():
    if False:
        for i in range(10):
            print('nop')
    transformed = (transform_dep_for_pip('python-for-android @ https://github.com/kivy/' + 'python-for-android/archive/master.zip'), transform_dep_for_pip('python-for-android @ https://github.com/kivy/' + 'python-for-android/archive/master.zip' + '#egg=python-for-android-master'), transform_dep_for_pip('python-for-android @ https://github.com/kivy/' + 'python-for-android/archive/master.zip' + '#'))
    expected = 'https://github.com/kivy/python-for-android/archive/master.zip' + '#egg=python-for-android'
    assert transformed == (expected, expected, expected)
    assert transform_dep_for_pip('https://a@b/') == 'https://a@b/'

def test_is_filesystem_path():
    if False:
        print('Hello World!')
    assert is_filesystem_path('/some/test')
    assert not is_filesystem_path('https://blubb')
    assert not is_filesystem_path('test @ bla')
    assert is_filesystem_path('/abc/c@d')
    assert not is_filesystem_path('https://user:pw@host/')
    assert is_filesystem_path('.')
    assert is_filesystem_path('')

def test_parse_as_folder_reference():
    if False:
        for i in range(10):
            print('nop')
    assert parse_as_folder_reference('file:///a%20test') == '/a test'
    assert parse_as_folder_reference('https://github.com') is None
    assert parse_as_folder_reference('/a/folder') == '/a/folder'
    assert parse_as_folder_reference('test @ /abc') == '/abc'
    assert parse_as_folder_reference('test @ https://bla') is None

class TestGetSystemPythonExecutable:
    """ This contains all tests for _get_system_python_executable().

    ULTRA IMPORTANT THING TO UNDERSTAND: (if you touch this)

    This code runs things with other python interpreters NOT in the tox
    environment/virtualenv.
    E.g. _get_system_python_executable() is outside in the regular
    host environment! That also means all dependencies can be possibly
    not present!

    This is kind of absurd that we need this to run the test at all,
    but we can't test this inside tox's virtualenv:
    """

    def test_basic(self):
        if False:
            while True:
                i = 10
        pybin = _get_system_python_executable()
        pyversion = subprocess.check_output([pybin, '-c', 'import sys; print(sys.version)'], stderr=subprocess.STDOUT).decode('utf-8', 'replace')
        assert pyversion.strip() == sys.version.strip()

    def run__get_system_python_executable(self, pybin):
        if False:
            for i in range(10):
                print('nop')
        ' Helper function to run our function.\n\n            We want to see what _get_system_python_executable() does given\n            a specific python, so we need to make it import it and run it,\n            with that TARGET python, which this function does.\n        '
        cmd = [pybin, '-c', "import importlib\nimport build.util\nimport os\nimport sys\nsys.path = [os.path.dirname(sys.argv[1])] + sys.path\nm = importlib.import_module(\n    os.path.basename(sys.argv[1]).partition('.')[0]\n)\nprint(m._get_system_python_executable())", os.path.join(os.path.dirname(__file__), '..', 'pythonforandroid', 'pythonpackage.py')]
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8', 'replace').strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError('call failed, with output: ' + str(e.output))

    def test_systemwide_python(self):
        if False:
            return 10
        pybin = _get_system_python_executable()
        try:
            p1 = os.path.normpath(self.run__get_system_python_executable(pybin))
            p2 = os.path.normpath(pybin)
            assert p1 == p2
        except RuntimeError as e:
            if 'build' in str(e.args):
                pass
            elif 'toml' in str(e.args):
                pass
            else:
                raise

    def test_venv(self):
        if False:
            i = 10
            return i + 15
        " Verifies that _get_system_python_executable() works correctly\n            in a 'venv' (Python 3 only feature).\n        "
        pybin = _get_system_python_executable()
        test_dir = tempfile.mkdtemp()
        try:
            subprocess.check_output([pybin, '-m', 'venv', '--', os.path.join(test_dir, 'venv')])
            subprocess.check_output([os.path.join(test_dir, 'venv', 'bin', 'pip'), 'install', '-U', 'pip'])
            subprocess.check_output([os.path.join(test_dir, 'venv', 'bin', 'pip'), 'install', '-U', 'build', 'toml', 'sh<2.0', 'colorama', 'appdirs', 'jinja2', 'packaging'])
            sys_python_path = self.run__get_system_python_executable(os.path.join(test_dir, 'venv', 'bin', 'python'))
            assert os.path.normpath(sys_python_path).startswith(os.path.normpath(pybin))
        finally:
            shutil.rmtree(test_dir)
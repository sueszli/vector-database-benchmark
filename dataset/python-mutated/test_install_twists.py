import os
import shutil
import sys
import pytest
from pipenv.utils.shell import temp_environ

@pytest.mark.extras
@pytest.mark.install
@pytest.mark.local
def test_local_path_issue_6016(pipenv_instance_pypi):
    if False:
        print('Hello World!')
    with pipenv_instance_pypi() as p:
        setup_py = os.path.join(p.path, 'setup.py')
        with open(setup_py, 'w') as fh:
            contents = "\nfrom setuptools import setup, find_packages\nsetup(\n    name='testpipenv',\n    version='0.1',\n    description='Pipenv Test Package',\n    author='Pipenv Test',\n    author_email='test@pipenv.package',\n    license='MIT',\n    packages=find_packages(),\n    install_requires=[],\n    extras_require={'dev': ['six']},\n    zip_safe=False\n)\n            ".strip()
            fh.write(contents)
        c = p.pipenv('install .')
        assert c.returncode == 0
        assert 'testpipenv' in p.lockfile['default']

@pytest.mark.extras
@pytest.mark.install
@pytest.mark.local
def test_local_extras_install(pipenv_instance_pypi):
    if False:
        while True:
            i = 10
    'Ensure -e .[extras] installs.\n    '
    with pipenv_instance_pypi() as p:
        setup_py = os.path.join(p.path, 'setup.py')
        with open(setup_py, 'w') as fh:
            contents = "\nfrom setuptools import setup, find_packages\nsetup(\n    name='testpipenv',\n    version='0.1',\n    description='Pipenv Test Package',\n    author='Pipenv Test',\n    author_email='test@pipenv.package',\n    license='MIT',\n    packages=find_packages(),\n    install_requires=[],\n    extras_require={'dev': ['six']},\n    zip_safe=False\n)\n            ".strip()
            fh.write(contents)
        line = '-e .[dev]'
        with open(os.path.join(p.path, 'Pipfile'), 'w') as fh:
            fh.write('\n[packages]\ntestpipenv = {path = ".", editable = true, extras = ["dev"]}\n\n[dev-packages]\n            '.strip())
        c = p.pipenv('install')
        assert c.returncode == 0
        assert 'testpipenv' in p.lockfile['default']
        assert p.lockfile['default']['testpipenv']['extras'] == ['dev']
        assert 'six' in p.lockfile['default']
        c = p.pipenv('uninstall --all')
        assert c.returncode == 0
        print(f'Current directory: {os.getcwd()}', file=sys.stderr)
        c = p.pipenv(f'install {line}')
        assert c.returncode == 0
        assert 'testpipenv' in p.pipfile['packages']
        assert p.pipfile['packages']['testpipenv']['file'] == '.'
        assert p.pipfile['packages']['testpipenv']['extras'] == ['dev']
        assert 'six' in p.lockfile['default']

@pytest.mark.local
@pytest.mark.install
@pytest.mark.needs_internet
class TestDirectDependencies:
    """Ensure dependency_links are parsed and installed.

    This is needed for private repo dependencies.
    """

    @staticmethod
    def helper_dependency_links_install_make_setup(pipenv_instance, deplink):
        if False:
            i = 10
            return i + 15
        setup_py = os.path.join(pipenv_instance.path, 'setup.py')
        with open(setup_py, 'w') as fh:
            contents = f"\nfrom setuptools import setup\n\nsetup(\n    name='testdeplinks',\n    version='0.1',\n    packages=[],\n    install_requires=[\n        '{deplink}'\n    ],\n)\n            ".strip()
            fh.write(contents)

    @staticmethod
    def helper_dependency_links_install_test(pipenv_instance, deplink):
        if False:
            return 10
        TestDirectDependencies.helper_dependency_links_install_make_setup(pipenv_instance, deplink)
        c = pipenv_instance.pipenv('install -v -e .')
        assert c.returncode == 0
        assert 'six' in pipenv_instance.lockfile['default']

    @pytest.mark.skip(reason='This test modifies os.environment which has side effects on other tests')
    def test_https_dependency_links_install(self, pipenv_instance_pypi):
        if False:
            return 10
        'Ensure dependency_links are parsed and installed (needed for private repo dependencies).\n        '
        with temp_environ(), pipenv_instance_pypi() as p:
            os.environ['PIP_NO_BUILD_ISOLATION'] = '1'
            TestDirectDependencies.helper_dependency_links_install_test(p, 'six@ git+https://github.com/benjaminp/six@1.11.0')

@pytest.mark.run
@pytest.mark.install
def test_normalize_name_install(pipenv_instance_private_pypi):
    if False:
        return 10
    with pipenv_instance_private_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = '\n# Pre comment\n[packages]\nRequests = "==2.14.0"   # Inline comment\n'
            f.write(contents)
        assert p.pipfile['packages']['Requests'] == '==2.14.0'
        c = p.pipenv('install requests==2.18.4')
        assert c.returncode == 0
        assert 'Requests' not in p.pipfile['packages']
        assert 'requests' in p.pipfile['packages']
        assert p.pipfile['packages']['requests'] == '==2.18.4'
        c = p.pipenv('install python_DateUtil')
        assert c.returncode == 0
        assert 'python-dateutil' in p.pipfile['packages']
        with open(p.pipfile_path) as f:
            contents = f.read()
            assert '# Pre comment' in contents

@pytest.mark.eggs
@pytest.mark.files
@pytest.mark.local
@pytest.mark.resolver
@pytest.mark.skip
def test_local_package(pipenv_instance_private_pypi, testsroot):
    if False:
        while True:
            i = 10
    'This test ensures that local packages (directories with a setup.py)\n    installed in editable mode have their dependencies resolved as well'
    file_name = 'requests-2.19.1.tar.gz'
    package = 'requests-2.19.1'
    source_path = os.path.abspath(os.path.join(testsroot, 'test_artifacts', file_name))
    with pipenv_instance_private_pypi() as p:
        copy_to = os.path.join(p.path, file_name)
        shutil.copy(source_path, copy_to)
        import tarfile
        with tarfile.open(copy_to, 'r:gz') as tgz:

            def is_within_directory(directory, target):
                if False:
                    return 10
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path='.', members=None, *, numeric_owner=False):
                if False:
                    while True:
                        i = 10
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception('Attempted Path Traversal in Tar File')
                tar.extractall(path, members, numeric_owner)
            safe_extract(tgz, path=p.path)
        c = p.pipenv(f'install -e {package}')
        assert c.returncode == 0
        assert all((pkg in p.lockfile['default'] for pkg in ['urllib3', 'idna', 'certifi', 'chardet']))

@pytest.mark.files
@pytest.mark.local
def test_local_tar_gz_file(pipenv_instance_private_pypi, testsroot):
    if False:
        i = 10
        return i + 15
    file_name = 'requests-2.19.1.tar.gz'
    with pipenv_instance_private_pypi() as p:
        requests_path = p._pipfile.get_fixture_path(f'{file_name}')
        c = p.pipenv(f'install {requests_path}')
        assert c.returncode == 0
        key = list(p.pipfile['packages'])[0]
        dep = p.pipfile['packages'][key]
        assert 'file' in dep or 'path' in dep
        assert c.returncode == 0
        dep = p.lockfile['default']['requests']
        assert 'file' in dep or 'path' in dep

@pytest.mark.urls
@pytest.mark.install
def test_install_local_uri_special_character(pipenv_instance_private_pypi, testsroot):
    if False:
        while True:
            i = 10
    file_name = 'six-1.11.0+mkl-py2.py3-none-any.whl'
    source_path = os.path.abspath(os.path.join(testsroot, 'test_artifacts', file_name))
    with pipenv_instance_private_pypi() as p:
        artifact_dir = 'artifacts'
        artifact_path = os.path.join(p.path, artifact_dir)
        os.makedirs(artifact_path, exist_ok=True)
        shutil.copy(source_path, os.path.join(artifact_path, file_name))
        with open(p.pipfile_path, 'w') as f:
            contents = f'\n# Pre comment\n[packages]\nsix = {{path = "./artifacts/{file_name}"}}\n            '
            f.write(contents.strip())
        c = p.pipenv('install')
        assert c.returncode == 0
        assert 'six' in p.lockfile['default']

@pytest.mark.run
@pytest.mark.files
@pytest.mark.install
def test_multiple_editable_packages_should_not_race(pipenv_instance_private_pypi, testsroot):
    if False:
        print('Hello World!')
    "Test for a race condition that can occur when installing multiple 'editable' packages at\n    once, and which causes some of them to not be importable.\n\n    This issue had been fixed for VCS packages already, but not local 'editable' packages.\n\n    So this test locally installs packages from tarballs that have already been committed in\n    the local `pypi` dir to avoid using VCS packages.\n    "
    pkgs = ['six', 'jinja2']
    with pipenv_instance_private_pypi() as p:
        pipfile_string = f'\n[[source]]\nurl = "{p.index_url}"\nverify_ssl = false\nname = "testindex"\n\n[dev-packages]\n\n[packages]\n        '
        for pkg_name in pkgs:
            source_path = p._pipfile.get_fixture_path(f'git/{pkg_name}/')
            shutil.copytree(source_path, pkg_name)
            pipfile_string += f'"{pkg_name}" = {{path = "./{pkg_name}", editable = true}}\n'
        with open(p.pipfile_path, 'w') as f:
            f.write(pipfile_string.strip())
        c = p.pipenv('install')
        assert c.returncode == 0
        c = p.pipenv('run python -c "import jinja2, six"')
        assert c.returncode == 0, c.stderr

@pytest.mark.outdated
def test_outdated_should_compare_postreleases_without_failing(pipenv_instance_private_pypi):
    if False:
        for i in range(10):
            print('nop')
    with pipenv_instance_private_pypi() as p:
        c = p.pipenv('install ibm-db-sa-py3==0.3.0')
        assert c.returncode == 0
        c = p.pipenv('update --outdated')
        assert c.returncode == 0
        assert 'Skipped Update' in c.stderr
        p._pipfile.update('ibm-db-sa-py3', '*')
        c = p.pipenv('update --outdated')
        assert c.returncode != 0
        assert 'out-of-date' in c.stdout

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='Package does not work with Python 3.12')
def test_install_remote_wheel_file_with_extras(pipenv_instance_pypi):
    if False:
        while True:
            i = 10
    with pipenv_instance_pypi() as p:
        c = p.pipenv('install fastapi[dev]@https://files.pythonhosted.org/packages/4e/1a/04887c641b67e6649bde845b9a631f73a7abfbe3afda83957e09b95d88eb/fastapi-0.95.2-py3-none-any.whl')
        assert c.returncode == 0
        assert 'ruff' in p.lockfile['default']
        assert 'pre-commit' in p.lockfile['default']
        assert 'uvicorn' in p.lockfile['default']

@pytest.mark.install
@pytest.mark.skip_lock
@pytest.mark.needs_internet
def test_install_skip_lock(pipenv_instance_private_pypi):
    if False:
        while True:
            i = 10
    with pipenv_instance_private_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = '\n[[source]]\nurl = "{}"\nverify_ssl = true\nname = "pypi"\n[packages]\nsix = {}\n            '.format(p.index_url, '{version = "*", index = "pypi"}').strip()
            f.write(contents)
        c = p.pipenv('install --skip-lock')
        assert c.returncode == 0
        c = p.pipenv('run python -c "import six"')
        assert c.returncode == 0
import os
import pytest
from flaky import flaky
from pipenv.project import Project
from pipenv.utils.shell import temp_environ

@pytest.mark.markers
def test_package_environment_markers(pipenv_instance_private_pypi):
    if False:
        for i in range(10):
            print('nop')
    with pipenv_instance_private_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = '\n[[source]]\nurl = "{}"\nverify_ssl = false\nname = "testindex"\n\n[packages]\nfake_package = {}\n\n[dev-packages]\n            '.format(p.index_url, '{version = "*", markers="os_name==\'splashwear\'", index="testindex"}').strip()
            f.write(contents)
        c = p.pipenv('install -v')
        assert c.returncode == 0
        assert 'markers' in p.lockfile['default']['fake_package'], p.lockfile['default']
        assert p.lockfile['default']['fake_package']['markers'] == "os_name == 'splashwear'"
        assert p.lockfile['default']['fake_package']['hashes'] == ['sha256:1531e01a7f306f496721f425c8404f3cfd8d4933ee6daf4668fcc70059b133f3', 'sha256:cf83dc3f6c34050d3360fbdf655b2652c56532e3028b1c95202611ba1ebdd624']
        c = p.pipenv('run python -c "import fake_package;"')
        assert c.returncode == 1

@flaky
@pytest.mark.markers
def test_platform_python_implementation_marker(pipenv_instance_private_pypi):
    if False:
        print('Hello World!')
    'Markers should be converted during locking to help users who input this\n    incorrectly.\n    '
    with pipenv_instance_private_pypi() as p:
        c = p.pipenv('install depends-on-marked-package')
        assert c.returncode == 0
        assert 'pytz' in p.lockfile['default']
        assert p.lockfile['default']['pytz'].get('markers') == "platform_python_implementation == 'CPython'"

@flaky
@pytest.mark.alt
@pytest.mark.markers
@pytest.mark.install
def test_specific_package_environment_markers(pipenv_instance_pypi):
    if False:
        for i in range(10):
            print('nop')
    with pipenv_instance_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = '\n[packages]\nsix = {version = "*", os_name = "== \'splashwear\'"}\n            '.strip()
            f.write(contents)
        c = p.pipenv('install')
        assert c.returncode == 0
        assert 'markers' in p.lockfile['default']['six']
        c = p.pipenv('run python -c "import six;"')
        assert c.returncode == 1

@flaky
@pytest.mark.markers
def test_top_level_overrides_environment_markers(pipenv_instance_pypi):
    if False:
        for i in range(10):
            print('nop')
    'Top-level environment markers should take precedence.\n    '
    with pipenv_instance_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = '\n[packages]\napscheduler = "*"\nfuncsigs = {version = "*", os_name = "== \'splashwear\'"}\n            '.strip()
            f.write(contents)
        c = p.pipenv('install')
        assert c.returncode == 0
        assert 'markers' in p.lockfile['default']['funcsigs'], p.lockfile['default']['funcsigs']
        assert p.lockfile['default']['funcsigs']['markers'] == "os_name == 'splashwear'", p.lockfile['default']['funcsigs']

@flaky
@pytest.mark.markers
@pytest.mark.install
def test_global_overrides_environment_markers(pipenv_instance_private_pypi):
    if False:
        return 10
    'Empty (unconditional) dependency should take precedence.\n    If a dependency is specified without environment markers, it should\n    override dependencies with environment markers. In this example,\n    APScheduler requires funcsigs only on Python 2, but since funcsigs is\n    also specified as an unconditional dep, its markers should be empty.\n    '
    with pipenv_instance_private_pypi() as p:
        with open(p.pipfile_path, 'w') as f:
            contents = f'\n[[source]]\nurl = "{p.index_url}"\nverify_ssl = false\nname = "testindex"\n\n[packages]\napscheduler = "*"\nfuncsigs = "*"\n            '.strip()
            f.write(contents)
        c = p.pipenv('install')
        assert c.returncode == 0
        assert p.lockfile['default']['funcsigs'].get('markers', '') == ''

@flaky
@pytest.mark.markers
@pytest.mark.complex
def test_resolver_unique_markers(pipenv_instance_pypi):
    if False:
        while True:
            i = 10
    'vcrpy has a dependency on `yarl` which comes with a marker\n    of \'python version in "3.4, 3.5, 3.6" - this marker duplicates itself:\n\n    \'yarl; python version in "3.4, 3.5, 3.6"; python version in "3.4, 3.5, 3.6"\'\n\n    This verifies that we clean that successfully.\n    '
    with pipenv_instance_pypi() as p:
        c = p.pipenv('install vcrpy==2.0.1')
        assert c.returncode == 0
        assert 'yarl' in p.lockfile['default']
        yarl = p.lockfile['default']['yarl']
        assert 'markers' in yarl
        assert yarl['markers'] in ["python_version in '3.4, 3.5, 3.6'", "python_version >= '3.4'", "python_version >= '3.5'"]

@flaky
@pytest.mark.project
@pytest.mark.needs_internet
def test_environment_variable_value_does_not_change_hash(pipenv_instance_private_pypi):
    if False:
        for i in range(10):
            print('nop')
    with pipenv_instance_private_pypi() as p, temp_environ():
        with open(p.pipfile_path, 'w') as f:
            f.write('\n[[source]]\nurl = \'https://${PYPI_USERNAME}:${PYPI_PASSWORD}@pypi.org/simple\'\nverify_ssl = true\nname = \'pypi\'\n\n[packages]\nsix = "*"\n')
        project = Project()
        os.environ['PYPI_USERNAME'] = 'whatever'
        os.environ['PYPI_PASSWORD'] = 'pass'
        assert project.get_lockfile_hash() is None
        c = p.pipenv('install')
        assert c.returncode == 0
        lock_hash = project.get_lockfile_hash()
        assert lock_hash is not None
        assert lock_hash == project.calculate_pipfile_hash()
        assert c.returncode == 0
        assert project.get_lockfile_hash() == project.calculate_pipfile_hash()
        os.environ['PYPI_PASSWORD'] = 'pass2'
        assert project.get_lockfile_hash() == project.calculate_pipfile_hash()
        with open(p.pipfile_path, 'a') as f:
            f.write('requests = "==2.14.0"\n')
        assert project.get_lockfile_hash() != project.calculate_pipfile_hash()
import os
import sys
import pytest

@pytest.mark.categories
@pytest.mark.install
def test_basic_category_install(pipenv_instance_private_pypi):
    if False:
        while True:
            i = 10
    with pipenv_instance_private_pypi() as p:
        c = p.pipenv('install six --categories prereq')
        assert c.returncode == 0
        assert 'six' not in p.pipfile['packages']
        assert 'six' not in p.lockfile['default']
        assert 'six' in p.pipfile['prereq']
        assert 'six' in p.lockfile['prereq']

@pytest.mark.categories
@pytest.mark.install
@pytest.mark.requirements
def test_basic_category_install_from_requirements(pipenv_instance_private_pypi):
    if False:
        return 10
    with pipenv_instance_private_pypi(pipfile=False) as p:
        with open('requirements.txt', 'w') as f:
            f.write('six==1.16.0')
        c = p.pipenv('install --categories prereq')
        assert c.returncode == 0
        os.unlink('requirements.txt')
        print(c.stdout)
        print(c.stderr)
        assert c.returncode == 0
        assert 'six' not in p.pipfile['packages']
        assert 'six' not in p.lockfile['default']
        assert 'six' in p.pipfile['prereq']
        assert 'six' in p.lockfile['prereq']

@pytest.mark.categories
@pytest.mark.install
@pytest.mark.parametrize('categories', ['prereq other', 'prereq, other'])
def test_multiple_category_install(pipenv_instance_private_pypi, categories):
    if False:
        print('Hello World!')
    with pipenv_instance_private_pypi() as p:
        c = p.pipenv('install six --categories="prereq other"')
        assert c.returncode == 0
        assert 'six' not in p.pipfile['packages']
        assert 'six' not in p.lockfile['default']
        assert 'six' in p.pipfile['prereq']
        assert 'six' in p.lockfile['prereq']
        assert 'six' in p.lockfile['other']
        assert 'six' in p.lockfile['other']

@pytest.mark.categories
@pytest.mark.install
@pytest.mark.requirements
def test_multiple_category_install_from_requirements(pipenv_instance_private_pypi):
    if False:
        for i in range(10):
            print('nop')
    with pipenv_instance_private_pypi(pipfile=False) as p:
        with open('requirements.txt', 'w') as f:
            f.write('six==1.16.0')
        c = p.pipenv('install --categories="prereq other"')
        assert c.returncode == 0
        os.unlink('requirements.txt')
        print(c.stdout)
        print(c.stderr)
        assert c.returncode == 0
        assert 'six' not in p.pipfile['packages']
        assert 'six' not in p.lockfile['default']
        assert 'six' in p.pipfile['prereq']
        assert 'six' in p.lockfile['prereq']
        assert 'six' in p.pipfile['other']
        assert 'six' in p.lockfile['other']

@pytest.mark.extras
@pytest.mark.install
@pytest.mark.local
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='test is not 3.12 compatible')
def test_multiple_category_install_proceeds_in_order_specified(pipenv_instance_private_pypi):
    if False:
        while True:
            i = 10
    'Ensure -e .[extras] installs.\n    '
    with pipenv_instance_private_pypi() as p:
        setup_py = os.path.join(p.path, 'setup.py')
        with open(setup_py, 'w') as fh:
            contents = "\nimport six\nfrom setuptools import setup\nsetup(\n    name='testpipenv',\n    version='0.1',\n    description='Pipenv Test Package',\n    author='Pipenv Test',\n    author_email='test@pipenv.package',\n    license='MIT',\n    packages=[],\n    install_requires=['six', 'setuptools'],\n    zip_safe=False\n)\n            ".strip()
            fh.write(contents)
        with open(os.path.join(p.path, 'Pipfile'), 'w') as fh:
            fh.write('\n[packages]\ntestpipenv = {path = ".", editable = true, skip_resolver = true}\n\n[prereq]\nsix = "*"\n            '.strip())
        c = p.pipenv('lock -v')
        assert c.returncode == 0
        assert 'testpipenv' in p.lockfile['default']
        assert 'testpipenv' not in p.lockfile['prereq']
        assert 'six' in p.lockfile['prereq']
        c = p.pipenv('sync --categories="prereq packages" --extra-pip-args="--no-build-isolation" -v')
        assert c.returncode == 0
import pytest
from test_pyenv_helpers import local_python_versions, not_installed_output, Native, Arch

def test_no_local_version(pyenv):
    if False:
        i = 10
        return i + 15
    assert pyenv.local() == ('no local version configured for this directory', '')

@pytest.mark.parametrize('settings', [lambda : {'local_ver': Native('3.8.9')}])
def test_local_version_defined(pyenv):
    if False:
        for i in range(10):
            print('nop')
    assert pyenv.local() == (Native('3.8.9'), '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.7.7'), Native('3.8.9')], 'local_ver': Native('3.8.9')}])
def test_local_set_installed_version(pyenv):
    if False:
        return 10
    assert pyenv.local(Arch('3.7.7')) == ('', '')
    assert pyenv.local() == (Native('3.7.7'), '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.8.9')]}])
def test_local_set_unknown_version(pyenv):
    if False:
        print('Hello World!')
    assert pyenv.local(Arch('3.7.8')) == (not_installed_output(Native('3.7.8')), '')

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.7.7'), Native('3.8.9')]}])
def test_local_set_many_versions(local_path, pyenv):
    if False:
        for i in range(10):
            print('nop')
    assert pyenv.local(Arch('3.7.7'), Arch('3.8.9')) == ('', '')
    assert local_python_versions(local_path) == '\n'.join([Native('3.7.7'), Native('3.8.9')])

@pytest.mark.parametrize('settings', [lambda : {'versions': [Native('3.7.7')]}])
def test_local_set_many_versions_one_not_installed(pyenv):
    if False:
        for i in range(10):
            print('nop')
    assert pyenv.local(Arch('3.7.7'), Arch('3.8.9')) == (not_installed_output(Native('3.8.9')), '')

@pytest.mark.parametrize('settings', [lambda : {'local_ver': [Native('3.7.7'), Native('3.8.9')]}])
def test_local_many_versions_defined(pyenv):
    if False:
        i = 10
        return i + 15
    assert pyenv.local() == ('\r\n'.join([Native('3.7.7'), Native('3.8.9')]), '')
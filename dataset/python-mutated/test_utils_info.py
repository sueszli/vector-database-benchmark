import os
import pytest
from featuretools import __version__
from featuretools.utils import get_featuretools_root, get_installed_packages, get_sys_info, show_info

@pytest.fixture
def this_dir():
    if False:
        while True:
            i = 10
    return os.path.dirname(os.path.abspath(__file__))

def test_show_info(capsys):
    if False:
        print('Hello World!')
    show_info()
    captured = capsys.readouterr()
    assert 'Featuretools version' in captured.out
    assert 'Featuretools installation directory:' in captured.out
    assert __version__ in captured.out
    assert 'SYSTEM INFO' in captured.out

def test_sys_info():
    if False:
        while True:
            i = 10
    sys_info = get_sys_info()
    info_keys = ['python', 'python-bits', 'OS', 'OS-release', 'machine', 'processor', 'byteorder', 'LC_ALL', 'LANG', 'LOCALE']
    found_keys = [k for (k, _) in sys_info]
    assert set(info_keys).issubset(found_keys)

def test_installed_packages():
    if False:
        i = 10
        return i + 15
    installed_packages = get_installed_packages()
    installed_set = {name.lower().replace('-', '_') for name in installed_packages.keys()}
    requirements = ['pandas', 'numpy', 'tqdm', 'cloudpickle', 'psutil']
    assert set(requirements).issubset(installed_set)

def test_get_featuretools_root(this_dir):
    if False:
        i = 10
        return i + 15
    root = os.path.abspath(os.path.join(this_dir, '..', '..'))
    assert get_featuretools_root() == root
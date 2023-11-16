import pytest
from astropy import test as run_tests

def test_module_not_found():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        run_tests(package='fake.module')

def test_pastebin_keyword():
    if False:
        return 10
    with pytest.raises(ValueError):
        run_tests(pastebin='not_an_option')

def test_unicode_literal_conversion():
    if False:
        while True:
            i = 10
    assert isinstance('ångström', str)
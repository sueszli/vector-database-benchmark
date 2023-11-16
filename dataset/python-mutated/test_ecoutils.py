import sys
from boltons import ecoutils

def test_basic():
    if False:
        for i in range(10):
            print('nop')
    prof = ecoutils.get_profile()
    assert prof['python']['bin'] == sys.executable

def test_scrub():
    if False:
        while True:
            i = 10
    prof = ecoutils.get_profile(scrub=True)
    assert prof['username'] == '-'
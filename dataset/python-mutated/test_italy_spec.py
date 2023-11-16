import re
import pytest
from mimesis.builtins import ItalySpecProvider
from mimesis.enums import Gender

@pytest.fixture
def italy():
    if False:
        for i in range(10):
            print('nop')
    return ItalySpecProvider()

def test_noun(italy):
    if False:
        while True:
            i = 10
    result = italy.fiscal_code(gender=Gender.MALE)
    assert re.fullmatch('^[A-Z]{6}\\d{2}[A-EHLMPR-T][0123][0-9][A-MZ]\\d{3}[A-Z]$', result)
    result = italy.fiscal_code(gender=Gender.FEMALE)
    assert re.fullmatch('^[A-Z]{6}\\d{2}[A-EHLMPR-T][4567][0-9][A-MZ]\\d{3}[A-Z]$', result)
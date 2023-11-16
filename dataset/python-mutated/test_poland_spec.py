import pytest
from mimesis.builtins.pl import PolandSpecProvider
from mimesis.enums import Gender
from mimesis.providers import Datetime

def validate_nip(nip):
    if False:
        while True:
            i = 10
    'Validate NIP.\n\n    :param nip: nip to validate\n    :return: True if nip is valid, False otherwise\n    '
    nip_digits = list(map(int, nip))
    args = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    sum_v = sum(map(lambda x: x[0] * x[1], zip(args, nip_digits)))
    checksum_digit = sum_v % 11
    return nip_digits[-1] == checksum_digit

def validate_pesel(pesel):
    if False:
        i = 10
        return i + 15
    'Validate PESEL.\n\n    :param pesel: pesel to validate\n    :return: True if pesel is valid, False otherwise\n    '
    pesel_digits = list(map(int, pesel))
    args = (9, 7, 3, 1, 9, 7, 3, 1, 9, 7)
    sum_v = sum(map(lambda x: x[0] * x[1], zip(args, pesel_digits)))
    return pesel_digits[-1] == sum_v % 10

def validate_regon(regon):
    if False:
        return 10
    'Validate REGON.\n\n    :param regon: regon to validate\n    :return: True if pesel is valid, False otherwise\n    '
    regon_digits = list(map(int, regon))
    args = (8, 9, 2, 3, 4, 5, 6, 7)
    sum_v = sum(map(lambda x: x[0] * x[1], zip(args, regon_digits)))
    checksum_digit = sum_v % 11
    if checksum_digit > 9:
        return regon_digits[-1] == 0
    return regon_digits[-1] == checksum_digit

@pytest.fixture
def pl():
    if False:
        while True:
            i = 10
    return PolandSpecProvider()

def test_nip(pl):
    if False:
        while True:
            i = 10
    nip = pl.nip()
    assert len(nip) == 10
    assert validate_nip(nip)

@pytest.mark.parametrize('gender', [Gender.FEMALE, Gender.MALE, None])
def test_pesel(pl, gender):
    if False:
        print('Hello World!')
    pesel = pl.pesel(gender=gender)
    assert len(pesel) == 11
    assert validate_pesel(pesel)
    birth_date = Datetime().datetime()
    pesel = pl.pesel(birth_date=birth_date, gender=Gender.MALE)
    assert len(pesel) == 11
    assert validate_pesel(pesel)

def test_regon(pl):
    if False:
        print('Hello World!')
    regon = pl.regon()
    assert len(regon) == 9
    assert validate_regon(regon)
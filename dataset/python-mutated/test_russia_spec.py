import pytest
from mimesis.builtins import RussiaSpecProvider
from mimesis.enums import Gender
from mimesis.exceptions import NonEnumerableError

@pytest.fixture
def russia():
    if False:
        return 10
    return RussiaSpecProvider()

def test_passport_series(russia):
    if False:
        i = 10
        return i + 15
    series = russia.passport_series()
    assert isinstance(series.split(' '), list)

def test_passport_series_parametrized(russia):
    if False:
        i = 10
        return i + 15
    series = russia.passport_series(year=10)
    (region, year) = series.split(' ')
    assert int(year) == 10
    assert 0 < int(region) < 100

def test_passport_number(russia):
    if False:
        while True:
            i = 10
    result = russia.passport_number()
    assert isinstance(result, int)
    assert result <= 999999 and result >= 100000

def test_series_and_number(russia):
    if False:
        i = 10
        return i + 15
    result = russia.series_and_number()
    assert result is not None

@pytest.mark.parametrize('gender', [Gender.FEMALE, Gender.MALE])
def test_patronymic(russia, gender):
    if False:
        print('Hello World!')
    result = russia.patronymic(gender=gender)
    assert result is not None
    assert len(result) >= 4
    with pytest.raises(NonEnumerableError):
        russia.patronymic(gender='nil')

def test_generate_sentence(russia):
    if False:
        print('Hello World!')
    result = russia.generate_sentence()
    assert len(result) >= 20
    assert isinstance(result, str)

def test_snils(russia):
    if False:
        while True:
            i = 10
    result = russia.snils()
    assert len(result) == 11

def test_inn(russia):
    if False:
        i = 10
        return i + 15
    result = russia.inn()
    assert isinstance(result, str)
    assert result is not None

def test_ogrn(russia):
    if False:
        while True:
            i = 10
    result = russia.ogrn()
    assert len(result) == 13

def test_bic(russia):
    if False:
        return 10
    result = russia.bic()
    assert len(result) == 9

def test_kpp(russia):
    if False:
        for i in range(10):
            print('nop')
    result = russia.kpp()
    assert len(result) == 9
import pytest
from mimesis.builtins import UkraineSpecProvider
from mimesis.enums import Gender
from mimesis.exceptions import NonEnumerableError

@pytest.fixture
def ukraine():
    if False:
        for i in range(10):
            print('nop')
    return UkraineSpecProvider()

@pytest.mark.parametrize('gender', [Gender.FEMALE, Gender.MALE])
def test_patronymic(ukraine, gender):
    if False:
        i = 10
        return i + 15
    result = ukraine.patronymic(gender=gender)
    assert result is not None
    assert len(result) >= 4
    with pytest.raises(NonEnumerableError):
        ukraine.patronymic(gender='nil')
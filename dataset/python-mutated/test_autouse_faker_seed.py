"""TEST NOTES:

The following tests cover cases where a ``faker_session_locale`` fixture was defined
by the user as well as a user-defined ``faker_seed`` autouse fixture. In this setup,
the plugin's ``DEFAULT_SEED`` will be ignored, and ``Faker`` instances will be seeded
using the value of ``faker_seed``. Said instances are still chosen in accordance to
how ``faker_locale`` and ``faker_session_locale`` interact with each other.
"""
from random import Random
import pytest
from tests.pytest.session_overrides.session_locale import _MODULE_LOCALES
_CHANGED_SEED = 4761

@pytest.fixture()
def faker_locale():
    if False:
        for i in range(10):
            print('nop')
    return ['it_IT']

@pytest.fixture(autouse=True)
def faker_seed():
    if False:
        for i in range(10):
            print('nop')
    return _CHANGED_SEED

def test_no_injection(_session_faker, faker):
    if False:
        i = 10
        return i + 15
    random = Random(_CHANGED_SEED)
    assert faker == _session_faker
    assert faker.locales == _MODULE_LOCALES
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()

def test_inject_faker_locale(_session_faker, faker, faker_locale):
    if False:
        while True:
            i = 10
    random = Random(_CHANGED_SEED)
    assert faker != _session_faker
    assert faker.locales == faker_locale
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()
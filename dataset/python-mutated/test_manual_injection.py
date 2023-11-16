"""TEST NOTES:

The following tests cover cases where a ``faker_session_locale`` fixture was defined
by the user as well as non-autouse ``faker_locale`` and ``faker_seed`` fixtures. The
resulting behavior of the ``faker`` fixture will vary depending on which fixtures
are injected.
"""
from random import Random
import pytest
from faker.contrib.pytest.plugin import DEFAULT_SEED
from tests.pytest.session_overrides.session_locale import _MODULE_LOCALES

@pytest.fixture()
def faker_locale():
    if False:
        i = 10
        return i + 15
    return ['it_IT']

@pytest.fixture()
def faker_seed():
    if False:
        while True:
            i = 10
    return 4761

def test_no_injection(_session_faker, faker):
    if False:
        while True:
            i = 10
    random = Random(DEFAULT_SEED)
    assert faker == _session_faker
    assert faker.locales == _MODULE_LOCALES
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()

def test_inject_faker_locale(_session_faker, faker, faker_locale):
    if False:
        while True:
            i = 10
    random = Random(DEFAULT_SEED)
    assert faker != _session_faker
    assert faker.locales == faker_locale
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()

def test_inject_faker_seed(_session_faker, faker, faker_seed):
    if False:
        print('Hello World!')
    random = Random(faker_seed)
    assert faker == _session_faker
    assert faker.locales == _MODULE_LOCALES
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()

def test_inject_faker_seed_and_locale(_session_faker, faker, faker_locale, faker_seed):
    if False:
        while True:
            i = 10
    random = Random(faker_seed)
    assert faker != _session_faker
    assert faker.locales == faker_locale
    assert faker.random != random
    assert faker.random.getstate() == random.getstate()
import pytest

@pytest.fixture
def fix1(fix2):
    if False:
        return 10
    return 1

@pytest.fixture
def fix2(fix1):
    if False:
        i = 10
        return i + 15
    return 1

def test(fix1):
    if False:
        while True:
            i = 10
    pass
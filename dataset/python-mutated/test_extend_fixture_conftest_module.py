import pytest

@pytest.fixture
def spam(spam):
    if False:
        i = 10
        return i + 15
    return spam * 2

def test_spam(spam):
    if False:
        for i in range(10):
            print('nop')
    assert spam == 'spamspam'
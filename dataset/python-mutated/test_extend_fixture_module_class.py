import pytest

@pytest.fixture
def spam():
    if False:
        return 10
    return 'spam'

class TestSpam:

    @pytest.fixture
    def spam(self, spam):
        if False:
            print('Hello World!')
        return spam * 2

    def test_spam(self, spam):
        if False:
            return 10
        assert spam == 'spamspam'
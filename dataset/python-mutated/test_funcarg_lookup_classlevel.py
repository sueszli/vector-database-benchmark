import pytest

class TestClass:

    @pytest.fixture
    def something(self, request):
        if False:
            print('Hello World!')
        return request.instance

    def test_method(self, something):
        if False:
            print('Hello World!')
        assert something is self
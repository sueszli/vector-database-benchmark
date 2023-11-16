import pytest

@pytest.mark.foo
def test_mark():
    if False:
        return 10
    pass
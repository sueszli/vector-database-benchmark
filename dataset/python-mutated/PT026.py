import pytest

@pytest.mark.usefixtures('a')
def test_ok():
    if False:
        for i in range(10):
            print('nop')
    pass

@pytest.mark.foo()
def test_ok_another_mark_with_parens():
    if False:
        while True:
            i = 10
    pass

@pytest.mark.foo
def test_ok_another_mark_no_parens():
    if False:
        print('Hello World!')
    pass

@pytest.mark.usefixtures()
def test_error_with_parens():
    if False:
        print('Hello World!')
    pass

@pytest.mark.usefixtures
def test_error_no_parens():
    if False:
        print('Hello World!')
    pass
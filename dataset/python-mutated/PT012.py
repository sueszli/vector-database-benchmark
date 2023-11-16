import pytest

def test_ok():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AttributeError):
        [].size

async def test_ok_trivial_with():
    with pytest.raises(AttributeError):
        with context_manager_under_test():
            pass
    with pytest.raises(ValueError):
        with context_manager_under_test():
            raise ValueError
    with pytest.raises(AttributeError):
        async with context_manager_under_test():
            pass

def test_ok_complex_single_call():
    if False:
        print('Hello World!')
    with pytest.raises(AttributeError):
        my_func([].size, [].size)

def test_ok_func_and_class():
    if False:
        return 10
    with pytest.raises(AttributeError):

        class A:
            pass
    with pytest.raises(AttributeError):

        def f():
            if False:
                i = 10
                return i + 15
            pass

def test_error_multiple_statements():
    if False:
        i = 10
        return i + 15
    with pytest.raises(AttributeError):
        len([])
        [].size

async def test_error_complex_statement():
    with pytest.raises(AttributeError):
        if True:
            [].size
    with pytest.raises(AttributeError):
        for i in []:
            [].size
    with pytest.raises(AttributeError):
        async for i in []:
            [].size
    with pytest.raises(AttributeError):
        while True:
            [].size
    with pytest.raises(AttributeError):
        async with context_manager_under_test():
            if True:
                raise Exception

def test_error_try():
    if False:
        while True:
            i = 10
    with pytest.raises(AttributeError):
        try:
            [].size
        except:
            raise
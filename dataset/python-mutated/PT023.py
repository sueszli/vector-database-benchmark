import pytest

@pytest.mark.foo(scope='module')
def ok_with_parameters_regardless_of_config():
    if False:
        print('Hello World!')
    pass

@pytest.mark.foo
def test_something():
    if False:
        return 10
    pass

@pytest.mark.foo
class TestClass:

    def test_something():
        if False:
            i = 10
            return i + 15
        pass

class TestClass:

    @pytest.mark.foo
    def test_something():
        if False:
            return 10
        pass

class TestClass:

    @pytest.mark.foo
    class TestNestedClass:

        def test_something():
            if False:
                return 10
            pass

class TestClass:

    class TestNestedClass:

        @pytest.mark.foo
        def test_something():
            if False:
                i = 10
                return i + 15
            pass

@pytest.mark.foo()
def test_something():
    if False:
        return 10
    pass

@pytest.mark.foo()
class TestClass:

    def test_something():
        if False:
            print('Hello World!')
        pass

class TestClass:

    @pytest.mark.foo()
    def test_something():
        if False:
            i = 10
            return i + 15
        pass

class TestClass:

    @pytest.mark.foo()
    class TestNestedClass:

        def test_something():
            if False:
                print('Hello World!')
            pass

class TestClass:

    class TestNestedClass:

        @pytest.mark.foo()
        def test_something():
            if False:
                for i in range(10):
                    print('nop')
            pass
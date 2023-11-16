from typing import Any, Callable
import pytest
from superset.utils.public_interfaces import compute_hash, get_warning_message
hashes: dict[Callable[..., Any], str] = {}

@pytest.mark.parametrize('interface,expected_hash', list(hashes.items()))
def test_public_interfaces(interface, expected_hash):
    if False:
        for i in range(10):
            print('nop')
    'Test that public interfaces have not been accidentally changed.'
    current_hash = compute_hash(interface)
    assert current_hash == expected_hash, get_warning_message(interface, current_hash)

def test_func_hash():
    if False:
        print('Hello World!')
    'Test that changing a function signature changes its hash.'

    def some_function(a, b):
        if False:
            print('Hello World!')
        return a + b
    original_hash = compute_hash(some_function)

    def some_function(a, b, c):
        if False:
            print('Hello World!')
        return a + b + c
    assert original_hash != compute_hash(some_function)

def test_class_hash():
    if False:
        for i in range(10):
            print('nop')
    'Test that changing a class changes its hash.'

    class SomeClass:

        def __init__(self, a, b):
            if False:
                print('Hello World!')
            self.a = a
            self.b = b

        def add(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.a + self.b
    original_hash = compute_hash(SomeClass)

    class SomeClass:

        def __init__(self, a, b, c):
            if False:
                print('Hello World!')
            self.a = a
            self.b = b
            self.c = c

        def add(self):
            if False:
                print('Hello World!')
            return self.a + self.b
    assert original_hash != compute_hash(SomeClass)

    class SomeClass:

        def __init__(self, a, b):
            if False:
                print('Hello World!')
            self.a = a
            self.b = b

        def sum(self):
            if False:
                while True:
                    i = 10
            return self.a + self.b
    assert original_hash != compute_hash(SomeClass)

    class SomeClass:

        def __init__(self, a, b):
            if False:
                for i in range(10):
                    print('nop')
            self.a = a
            self.b = b

        def add(self):
            if False:
                i = 10
                return i + 15
            return self._sum()

        def _sum(self):
            if False:
                print('Hello World!')
            return self.a + self.b
    assert original_hash == compute_hash(SomeClass)
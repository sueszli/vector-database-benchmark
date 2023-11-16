from typing import Optional, Union
from strawberry.annotation import StrawberryAnnotation

def test_bool():
    if False:
        while True:
            i = 10
    annotation = StrawberryAnnotation(bool)
    resolved = annotation.resolve()
    assert resolved is bool

def test_float():
    if False:
        print('Hello World!')
    annotation = StrawberryAnnotation(float)
    resolved = annotation.resolve()
    assert resolved is float

def test_int():
    if False:
        while True:
            i = 10
    annotation = StrawberryAnnotation(int)
    resolved = annotation.resolve()
    assert resolved is int

def test_str():
    if False:
        while True:
            i = 10
    annotation = StrawberryAnnotation(str)
    resolved = annotation.resolve()
    assert resolved is str

def test_none():
    if False:
        print('Hello World!')
    annotation = StrawberryAnnotation(None)
    annotation.resolve()
    annotation = StrawberryAnnotation(type(None))
    annotation.resolve()
    annotation = StrawberryAnnotation(Optional[int])
    annotation.resolve()
    annotation = StrawberryAnnotation(Union[None, int])
    annotation.resolve()
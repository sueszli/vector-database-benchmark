""" Pep-0484 type hinting """

class A:
    pass

def function_parameters(a: A, b, c: str, d: int, e: str, f: str, g: int=4):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param e: if docstring and annotation agree, only one should be returned\n    :type e: str\n    :param f: if docstring and annotation disagree, both should be returned\n    :type f: int\n    '
    a
    b
    c
    d
    e
    f
    g

def return_unspecified():
    if False:
        print('Hello World!')
    pass
return_unspecified()

def return_none() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Return type None means the same as no return type as far as jedi\n    is concerned\n    '
    pass
return_none()

def return_str() -> str:
    if False:
        i = 10
        return i + 15
    pass
return_str()

def return_custom_class() -> A:
    if False:
        print('Hello World!')
    pass
return_custom_class()

def return_annotation_and_docstring() -> str:
    if False:
        return 10
    '\n    :rtype: int\n    '
    pass
return_annotation_and_docstring()

def return_annotation_and_docstring_different() -> str:
    if False:
        while True:
            i = 10
    '\n    :rtype: str\n    '
    pass
return_annotation_and_docstring_different()

def annotation_forward_reference(b: 'B') -> 'B':
    if False:
        i = 10
        return i + 15
    b
annotation_forward_reference(1).t

class B:
    test_element = 1
    pass
annotation_forward_reference(1)

class SelfReference:
    test_element = 1

    def test_method(self, x: 'SelfReference') -> 'SelfReference':
        if False:
            return 10
        x
        self.t
        x.t
        self.test_method(1).t
SelfReference().test_method()

def function_with_non_pep_0484_annotation(x: 'I can put anything here', xx: '', yy: '\r\n\x00;+*&^564835(---^&*34', y: 3 + 3, zz: float) -> int('42'):
    if False:
        i = 10
        return i + 15
    x
    xx
    yy
    y
    zz
function_with_non_pep_0484_annotation(1, 2, 3, 'force string')

def function_forward_reference_dynamic(x: return_str_type(), y: 'return_str_type()') -> None:
    if False:
        for i in range(10):
            print('nop')
    x
    y

def return_str_type():
    if False:
        return 10
    return str
X = str

def function_with_assined_class_in_reference(x: X, y: 'Y'):
    if False:
        for i in range(10):
            print('nop')
    x
    y
Y = int

def just_because_we_can(x: 'flo' + 'at'):
    if False:
        while True:
            i = 10
    x

def keyword_only(a: str, *, b: str):
    if False:
        return 10
    a.startswi
    b.startswi

def argskwargs(*args: int, **kwargs: float):
    if False:
        while True:
            i = 10
    '\n    This might be a bit confusing, but is part of the standard.\n    args is changed to Tuple[int] in this case and kwargs to Dict[str, float],\n    which makes sense if you think about it a bit.\n    '
    args
    args[0]
    next(iter(kwargs.keys()))
    kwargs['']

class NotCalledClass:

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.x: int = x
        self.y: int = ''
        self.x
        self.y
        self.y
        self.z: int
        self.z = ''
        self.z
        self.w: float
        self.w
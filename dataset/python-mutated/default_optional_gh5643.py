try:
    from typing import Optional
except ImportError:
    pass

def gh5643_optional(a: Optional[int]=None):
    if False:
        while True:
            i = 10
    '\n    >>> gh5643_optional()\n    True\n    >>> gh5643_optional(1)\n    False\n    '
    return a is None

def gh5643_int_untyped(a: int=1, b=None):
    if False:
        print('Hello World!')
    '\n    >>> gh5643_int_untyped(2)\n    (False, True)\n    >>> gh5643_int_untyped(2, None)\n    (False, True)\n    >>> gh5643_int_untyped(1, 3)\n    (True, False)\n    '
    return (a == 1, b is None)

def gh5643_int_int_none(a: int=1, b: int=None):
    if False:
        print('Hello World!')
    '\n    >>> gh5643_int_int_none()\n    (True, True)\n    >>> gh5643_int_int_none(2, 3)\n    (False, False)\n    '
    return (a == 1, b is None)

def gh5643_int_int_integer(a: int=1, b: int=3):
    if False:
        i = 10
        return i + 15
    '\n    >>> gh5643_int_int_integer()\n    (True, True)\n    >>> gh5643_int_int_integer(2, 3)\n    (False, True)\n    '
    return (a == 1, b == 3)

def gh5643_int_optional_none(a: int=1, b: Optional[int]=None):
    if False:
        i = 10
        return i + 15
    '\n    >>> gh5643_int_optional_none()\n    (True, True)\n    >>> gh5643_int_optional_none(2)\n    (False, True)\n    >>> gh5643_int_optional_none(2, 3)\n    (False, False)\n    '
    return (a == 1, b is None)

def gh5643_int_optional_integer(a: int=1, b: Optional[int]=2):
    if False:
        print('Hello World!')
    '\n    >>> gh5643_int_optional_integer()\n    (True, True)\n    >>> gh5643_int_optional_integer(2)\n    (False, True)\n    >>> gh5643_int_optional_integer(2, 3)\n    (False, False)\n    >>> gh5643_int_optional_integer(2, 2)\n    (False, True)\n    '
    return (a == 1, b == 2)
_WARNINGS = "\n37:36: PEP-484 recommends 'typing.Optional[...]' for arguments that can be None.\n"
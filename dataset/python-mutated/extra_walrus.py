import cython
import sys

@cython.test_assert_path_exists('//PythonCapiCallNode')
def optimized(x):
    if False:
        for i in range(10):
            print('nop')
    "\n    x*2 is optimized to a PythonCapiCallNode. The test fails unless the CloneNode is kept up-to-date\n    (in the event that the optimization changes and test_assert_path_exists fails, the thing to do\n    is to find another case that's similarly optimized - the test isn't specifically interested in\n    multiplication)\n\n    >>> optimized(5)\n    10\n    "
    return (x := (x * 2))

@cython.test_fail_if_path_exists('//CloneNode')
def optimize_literals1():
    if False:
        return 10
    "\n    There's a small optimization for literals to avoid creating unnecessary temps\n    >>> optimize_literals1()\n    10\n    "
    x = 5
    return (x := 10)

@cython.test_fail_if_path_exists('//CloneNode')
def optimize_literals2():
    if False:
        for i in range(10):
            print('nop')
    "\n    There's a small optimization for literals to avoid creating unnecessary temps\n\n    >>> optimize_literals2()\n    'a string'\n    "
    x = 5
    return (x := u'a string')

@cython.test_fail_if_path_exists('//CloneNode')
def optimize_literals3():
    if False:
        i = 10
        return i + 15
    "\n    There's a small optimization for literals to avoid creating unnecessary temps\n\n    >>> optimize_literals3()\n    b'a bytes'\n    "
    x = 5
    return (x := b'a bytes')

@cython.test_fail_if_path_exists('//CloneNode')
def optimize_literals4():
    if False:
        while True:
            i = 10
    "\n    There's a small optimization for literals to avoid creating unnecessary temps\n\n    >>> optimize_literals4()\n    ('tuple', 1, 1.0, b'stuff')\n    "
    x = 5
    return (x := (u'tuple', 1, 1.0, b'stuff'))

@cython.test_fail_if_path_exists('//CoerceToPyTypeNode//AssignmentExpressionNode')
def avoid_extra_coercion(x: cython.double):
    if False:
        for i in range(10):
            print('nop')
    '\n    The assignment expression and x are both coerced to PyObject - this should happen only once\n    rather than to both separately\n    >>> avoid_extra_coercion(5.)\n    5.0\n    '
    y: object = "I'm an object"
    return (y := x)

async def async_func():
    """
    DW doesn't understand async functions well enough to make it a runtime test, but it was causing
    a compile-time failure at one point
    """
    if (variable := 1):
        pass
y_global = 6

class InLambdaInClass:
    """
    >>> InLambdaInClass.x1
    12
    >>> InLambdaInClass.x2
    [12, 12]
    """
    x1 = (lambda y_global: (y_global := (y_global + 1)) + y_global)(2) + y_global
    x2 = [(lambda y_global: (y_global := (y_global + 1)) + y_global)(2) + y_global for _ in range(2)]

def in_lambda_in_list_comprehension1():
    if False:
        i = 10
        return i + 15
    '\n    >>> in_lambda_in_list_comprehension1()\n    [[0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6]]\n    '
    return [(lambda x: [(x := y) + x for y in range(4)])(x) for x in range(5)]

def in_lambda_in_list_comprehension2():
    if False:
        print('Hello World!')
    '\n    >>> in_lambda_in_list_comprehension2()\n    [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]\n    '
    return [(lambda z: [(x := y) + z for y in range(4)])(x) for x in range(5)]

def in_lambda_in_generator_expression1():
    if False:
        print('Hello World!')
    '\n    >>> in_lambda_in_generator_expression1()\n    [(0, 2, 4, 6), (0, 2, 4, 6), (0, 2, 4, 6), (0, 2, 4, 6), (0, 2, 4, 6)]\n    '
    return [(lambda x: tuple(((x := y) + x for y in range(4))))(x) for x in range(5)]

def in_lambda_in_generator_expression2():
    if False:
        i = 10
        return i + 15
    '\n    >>> in_lambda_in_generator_expression2()\n    [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)]\n    '
    return [(lambda z: tuple(((x := y) + z for y in range(4))))(x) for x in range(5)]
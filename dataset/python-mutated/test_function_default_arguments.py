"""Default Argument Values

@see: https://docs.python.org/3/tutorial/controlflow.html#default-argument-values

The most useful form is to specify a default value for one or more arguments. This creates a
function that can be called with fewer arguments than it is defined to allow.
"""

def power_of(number, power=2):
    if False:
        i = 10
        return i + 15
    ' Raises number to specific power.\n\n    You may notice that by default the function raises number to the power of two.\n    '
    return number ** power

def test_default_function_arguments():
    if False:
        while True:
            i = 10
    'Test default function arguments'
    assert power_of(3) == 9
    assert power_of(3, 2) == 9
    assert power_of(3, 3) == 27
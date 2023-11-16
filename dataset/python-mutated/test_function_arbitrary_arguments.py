"""Arbitrary Argument Lists

@see: https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists

Function can be called with an arbitrary number of arguments. These arguments will be wrapped up in
a tuple. Before the variable number of arguments, zero or more normal arguments may occur.
"""

def test_function_arbitrary_arguments():
    if False:
        for i in range(10):
            print('nop')
    'Arbitrary Argument Lists'

    def test_function(first_param, *arguments):
        if False:
            for i in range(10):
                print('nop')
        'This function accepts its arguments through "arguments" tuple'
        assert first_param == 'first param'
        assert arguments == ('second param', 'third param')
    test_function('first param', 'second param', 'third param')

    def concat(*args, sep='/'):
        if False:
            for i in range(10):
                print('nop')
        return sep.join(args)
    assert concat('earth', 'mars', 'venus') == 'earth/mars/venus'
    assert concat('earth', 'mars', 'venus', sep='.') == 'earth.mars.venus'
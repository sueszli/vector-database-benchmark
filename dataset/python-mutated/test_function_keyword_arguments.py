"""Keyword Arguments

@see: https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments

Functions can be called using keyword arguments of the form kwarg=value.
"""
import pytest

def parrot(voltage, state='a stiff', action='voom', parrot_type='Norwegian Blue'):
    if False:
        while True:
            i = 10
    'Example of multi-argument function\n\n    This function accepts one required argument (voltage) and three optional arguments\n    (state, action, and type).\n    '
    message = "This parrot wouldn't " + action + ' '
    message += 'if you put ' + str(voltage) + ' volts through it. '
    message += 'Lovely plumage, the ' + parrot_type + '. '
    message += "It's " + state + '!'
    return message

def test_function_keyword_arguments():
    if False:
        for i in range(10):
            print('nop')
    'Test calling function with specifying keyword arguments'
    message = "This parrot wouldn't voom if you put 1000 volts through it. Lovely plumage, the Norwegian Blue. It's a stiff!"
    assert parrot(1000) == message
    assert parrot(voltage=1000) == message
    message = "This parrot wouldn't VOOOOOM if you put 1000000 volts through it. Lovely plumage, the Norwegian Blue. It's a stiff!"
    assert parrot(voltage=1000000, action='VOOOOOM') == message
    assert parrot(action='VOOOOOM', voltage=1000000) == message
    message = "This parrot wouldn't jump if you put 1000000 volts through it. Lovely plumage, the Norwegian Blue. It's bereft of life!"
    assert parrot(1000000, 'bereft of life', 'jump') == message
    message = "This parrot wouldn't voom if you put 1000 volts through it. Lovely plumage, the Norwegian Blue. It's pushing up the daisies!"
    assert parrot(1000, state='pushing up the daisies') == message
    with pytest.raises(Exception):
        parrot()
    with pytest.raises(Exception):
        parrot(110, voltage=220)
    with pytest.raises(Exception):
        parrot(actor='John Cleese')

    def function_with_one_argument(number):
        if False:
            i = 10
            return i + 15
        return number
    with pytest.raises(Exception):
        function_with_one_argument(0, number=0)

    def test_function(first_param, *arguments, **keywords):
        if False:
            for i in range(10):
                print('nop')
        'This function accepts its arguments through "arguments" tuple and keywords dictionary.'
        assert first_param == 'first param'
        assert arguments == ('second param', 'third param')
        assert keywords == {'fourth_param_name': 'fourth named param', 'fifth_param_name': 'fifth named param'}
    test_function('first param', 'second param', 'third param', fourth_param_name='fourth named param', fifth_param_name='fifth named param')
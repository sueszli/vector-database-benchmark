"""Unpacking Argument Lists

@see: https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists

Unpacking arguments may be executed via * and ** operators. See below for further details.
"""

def test_function_unpacking_arguments():
    if False:
        for i in range(10):
            print('nop')
    'Unpacking Argument Lists'
    assert list(range(3, 6)) == [3, 4, 5]
    arguments_list = [3, 6]
    assert list(range(*arguments_list)) == [3, 4, 5]

    def function_that_receives_names_arguments(first_word, second_word):
        if False:
            i = 10
            return i + 15
        return first_word + ', ' + second_word + '!'
    arguments_dictionary = {'first_word': 'Hello', 'second_word': 'World'}
    assert function_that_receives_names_arguments(**arguments_dictionary) == 'Hello, World!'
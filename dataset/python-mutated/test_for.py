"""FOR statement

@see: https://docs.python.org/3/tutorial/controlflow.html

The for statement in Python differs a bit from what you may be used to in C or Pascal.
Rather than always iterating over an arithmetic progression of numbers (like in Pascal), or
giving the user the ability to define both the iteration step and halting condition (as C),
Python’s for statement iterates over the items of any sequence (a list or a string), in the
order that they appear in the sequence. For example (no pun intended):
"""

def test_for_statement():
    if False:
        print('Hello World!')
    'FOR statement'
    words = ['cat', 'window', 'defenestrate']
    words_length = 0
    for word in words:
        words_length += len(word)
    assert words_length == 3 + 6 + 12
    for word in words[:]:
        if len(word) > 6:
            words.insert(0, word)
    assert words == ['defenestrate', 'cat', 'window', 'defenestrate']
    iterated_numbers = []
    for number in range(5):
        iterated_numbers.append(number)
    assert iterated_numbers == [0, 1, 2, 3, 4]
    words = ['Mary', 'had', 'a', 'little', 'lamb']
    concatenated_string = ''
    for word_index in range(len(words)):
        concatenated_string += words[word_index] + ' '
    assert concatenated_string == 'Mary had a little lamb '
    concatenated_string = ''
    for (word_index, word) in enumerate(words):
        concatenated_string += word + ' '
    assert concatenated_string == 'Mary had a little lamb '
    knights_names = []
    knights_properties = []
    knights = {'gallahad': 'the pure', 'robin': 'the brave'}
    for (key, value) in knights.items():
        knights_names.append(key)
        knights_properties.append(value)
    assert knights_names == ['gallahad', 'robin']
    assert knights_properties == ['the pure', 'the brave']
    indices = []
    values = []
    for (index, value) in enumerate(['tic', 'tac', 'toe']):
        indices.append(index)
        values.append(value)
    assert indices == [0, 1, 2]
    assert values == ['tic', 'tac', 'toe']
    questions = ['name', 'quest', 'favorite color']
    answers = ['lancelot', 'the holy grail', 'blue']
    combinations = []
    for (question, answer) in zip(questions, answers):
        combinations.append('What is your {0}?  It is {1}.'.format(question, answer))
    assert combinations == ['What is your name?  It is lancelot.', 'What is your quest?  It is the holy grail.', 'What is your favorite color?  It is blue.']

def test_range_function():
    if False:
        i = 10
        return i + 15
    'Range function\n\n    If you do need to iterate over a sequence of numbers, the built-in function range() comes in\n    handy. It generates arithmetic progressions.\n\n    In many ways the object returned by range() behaves as if it is a list, but in fact it isn’t.\n    It is an object which returns the successive items of the desired sequence when you iterate\n    over it, but it doesn’t really make the list, thus saving space.\n\n    We say such an object is iterable, that is, suitable as a target for functions and constructs\n    that expect something from which they can obtain successive items until the supply is exhausted.\n    We have seen that the for statement is such an iterator. The function list() is another; it\n    creates lists from iterables:\n    '
    assert list(range(5)) == [0, 1, 2, 3, 4]
    assert list(range(5, 10)) == [5, 6, 7, 8, 9]
    assert list(range(0, 10, 3)) == [0, 3, 6, 9]
    assert list(range(-10, -100, -30)) == [-10, -40, -70]
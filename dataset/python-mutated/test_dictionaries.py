"""Dictionaries.

@see: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
@see: https://www.w3schools.com/python/python_dictionaries.asp

A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are
written with curly brackets, and they have keys and values.

Dictionaries are sometimes found in other languages as “associative memories” or “associative
arrays”. Unlike sequences, which are indexed by a range of numbers, dictionaries are indexed by
keys, which can be any immutable type; strings and numbers can always be keys. Tuples can be used
as keys if they contain only strings, numbers, or tuples; if a tuple contains any mutable object
either directly or indirectly, it cannot be used as a key. You can’t use lists as keys, since
lists can be modified in place using index assignments, slice assignments, or methods like append()
and extend().

It is best to think of a dictionary as a set of key: value pairs, with the requirement that the
keys are unique (within one dictionary). A pair of braces creates an empty dictionary: {}.
Placing a comma-separated list of key:value pairs within the braces adds initial key:value pairs
to the dictionary; this is also the way dictionaries are written on output.
"""

def test_dictionary():
    if False:
        return 10
    'Dictionary'
    fruits_dictionary = {'cherry': 'red', 'apple': 'green', 'banana': 'yellow'}
    assert isinstance(fruits_dictionary, dict)
    assert fruits_dictionary['apple'] == 'green'
    assert fruits_dictionary['banana'] == 'yellow'
    assert fruits_dictionary['cherry'] == 'red'
    assert 'apple' in fruits_dictionary
    assert 'pineapple' not in fruits_dictionary
    fruits_dictionary['apple'] = 'red'
    fruits_dictionary['pineapple'] = 'yellow'
    assert fruits_dictionary['pineapple'] == 'yellow'
    assert list(fruits_dictionary) == ['cherry', 'apple', 'banana', 'pineapple']
    assert sorted(fruits_dictionary) == ['apple', 'banana', 'cherry', 'pineapple']
    del fruits_dictionary['pineapple']
    assert list(fruits_dictionary) == ['cherry', 'apple', 'banana']
    dictionary_via_constructor = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
    assert dictionary_via_constructor['sape'] == 4139
    assert dictionary_via_constructor['guido'] == 4127
    assert dictionary_via_constructor['jack'] == 4098
    dictionary_via_expression = {x: x ** 2 for x in (2, 4, 6)}
    assert dictionary_via_expression[2] == 4
    assert dictionary_via_expression[4] == 16
    assert dictionary_via_expression[6] == 36
    dictionary_for_string_keys = dict(sape=4139, guido=4127, jack=4098)
    assert dictionary_for_string_keys['sape'] == 4139
    assert dictionary_for_string_keys['guido'] == 4127
    assert dictionary_for_string_keys['jack'] == 4098
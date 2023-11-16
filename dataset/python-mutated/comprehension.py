"""
This module shows one-liner comprehensions where we make lists, tuples,
sets and dictionaries by looping through iterators.
"""

def main():
    if False:
        print('Hello World!')
    assert [0 for _ in range(5)] == [0] * 5 == [0, 0, 0, 0, 0]
    words = ['cat', 'mice', 'horse', 'bat']
    tuple_comp = tuple((len(word) for word in words))
    assert tuple_comp == (3, 4, 5, 3)
    set_comp = {len(word) for word in words}
    assert len(set_comp) < len(words)
    assert set_comp == {3, 4, 5}
    dict_comp = {word: len(word) for word in words}
    assert len(dict_comp) == len(words)
    assert dict_comp == {'cat': 3, 'mice': 4, 'horse': 5, 'bat': 3}
    nums = [31, 13, 64, 12, 767, 84]
    odds = [_ for _ in nums if _ % 2 == 1]
    assert odds == [31, 13, 767]
if __name__ == '__main__':
    main()